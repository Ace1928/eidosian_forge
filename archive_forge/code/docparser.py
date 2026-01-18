import ast
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import logging
from typing import List, Dict, Any, Optional, Union
import docstring_parser

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="docparser.log",
    filemode="w",
)


class TypeInferenceError(Exception):
    """Custom exception for type inference errors."""

    pass


class CodeParser(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.classes: List[Dict[str, Any]] = []
        logging.info("CodeParser initialized with an empty list of classes.")

    def visit_ClassDef(self, node: ast.ClassDef):
        logging.debug(f"Visiting class definition: {node.name}")
        description: str = (
            ast.get_docstring(node, clean=True) or "Description not provided."
        )
        class_info: Dict[str, Any] = {
            "ClassInformation": {
                "ClassName": node.name,
                "Description": description.strip(),
                "Version": "1.0",
            },
            "Methods": [],
            "Properties": self.extract_properties(node),
            "Dependencies": self.extract_dependencies(node),
        }
        self.classes.append(class_info)
        logging.info(
            f"Class {node.name} has been successfully added with its properties and methods."
        )
        self.generic_visit(node)

    def extract_properties(self, node: ast.ClassDef) -> List[Dict[str, str]]:
        properties: List[Dict[str, str]] = []
        for assign in node.body:
            if isinstance(assign, ast.AnnAssign):
                property_info: Dict[str, str] = {
                    "Name": assign.target.id,
                    "Type": self.infer_type(assign) or "Type not determined",
                    "Description": ast.get_docstring(assign)
                    or "No property description available.",
                }
                properties.append(property_info)
                logging.debug(f"Property extracted: {property_info}")
        logging.debug(f"Properties extracted from {node.name}: {properties}")
        return properties

    def parse_docstring(self, docstring: str):
        parsed = docstring_parser.parse(docstring)
        return {
            "Short Description": parsed.short_description,
            "Long Description": parsed.long_description,
            "Parameters": [
                {"Name": p.arg_name, "Type": p.type_name, "Description": p.description}
                for p in parsed.params
            ],
            "Returns": (
                {
                    "Type": parsed.returns.type_name,
                    "Description": parsed.returns.description,
                }
                if parsed.returns
                else None
            ),
            "Raises": [
                {"Type": ex.type_name, "Description": ex.description}
                for ex in parsed.raises
            ],
            "Examples": parsed.examples or "No examples provided.",
            "See Also": parsed.see_also or "No additional references.",
        }

    def enhance_method_info(self, method_info, docstring):
        enhanced_doc = self.parse_docstring(docstring)
        method_info.update(enhanced_doc)

    def infer_type(self, assign: Union[ast.AnnAssign, ast.arg]) -> str:
        """
        Infers the type of a given assignment or argument node in the abstract syntax tree (AST).
        This method handles various types of annotations including simple names, attributes, and subscripts.
        It is designed to be robust, handling nested types and providing detailed error logging.

        Parameters:
            assign (Union[ast.AnnAssign, ast.arg]): The AST node for which the type is to be inferred.

        Returns:
            str: The inferred type as a string. If the type cannot be determined, returns 'Unknown'.

        Raises:
            TypeInferenceError: If an error occurs during type inference that prevents completion.
        """
        try:
            if isinstance(assign, ast.AnnAssign):
                return self._infer_ann_assign_type(assign)
            elif isinstance(assign, ast.arg):
                return self._infer_arg_type(assign)
        except Exception as e:
            logging.error(f"Error inferring type: {e}")
            raise TypeInferenceError(
                f"An error occurred while inferring type: {str(e)}"
            )

    def _infer_ann_assign_type(self, assign: ast.AnnAssign) -> str:
        """
        Helper method to infer type from an AnnAssign node.

        Parameters:
            assign (ast.AnnAssign): The AnnAssign node from AST.

        Returns:
            str: The inferred type as a string.
        """
        if isinstance(assign.annotation, ast.Name):
            return assign.annotation.id
        elif isinstance(assign.annotation, ast.Attribute):
            return f"{assign.annotation.value.id}.{assign.annotation.attr}"
        elif isinstance(assign.annotation, ast.Subscript):
            base = self.infer_type(
                assign.annotation.value
            )  # Recursively infer the base type
            if isinstance(assign.annotation.slice, ast.Index):
                index = self.infer_type(assign.annotation.slice.value)
                return f"{base}[{index}]"
        return "Unknown"

    def _infer_arg_type(self, assign: ast.arg) -> str:
        """
        Helper method to infer type from an arg node.

        Parameters:
            assign (ast.arg): The arg node from AST.

        Returns:
            str: The inferred type as a string.
        """
        if assign.annotation:
            if isinstance(assign.annotation, ast.Name):
                return assign.annotation.id
            elif isinstance(assign.annotation, ast.Attribute):
                return f"{assign.annotation.value.id}.{assign.annotation.attr}"
            elif isinstance(assign.annotation, ast.Subscript):
                base = self.infer_type(assign.annotation.value)
                if isinstance(assign.annotation.slice, ast.Index):
                    index = self.infer_type(assign.annotation.slice.value)
                    return f"{base}[{index}]"
        return "Unknown"

    def extract_dependencies(self, node: ast.ClassDef) -> List[str]:
        dependencies: List[str] = []
        for body_item in node.body:
            if isinstance(body_item, (ast.Import, ast.ImportFrom)):
                if isinstance(body_item, ast.Import):
                    dependencies.extend([alias.name for alias in body_item.names])
                elif isinstance(body_item, ast.ImportFrom):
                    dependencies.extend(
                        [
                            f"{body_item.module}.{alias.name}"
                            for alias in body_item.names
                        ]
                    )
        logging.debug(f"Dependencies extracted from {node.name}: {dependencies}")
        return dependencies

    def visit_FunctionDef(self, node: ast.FunctionDef):
        logging.debug(f"Visiting function definition: {node.name}")
        docstring: str = (
            ast.get_docstring(node, clean=True) or "Function description not provided."
        )
        method_info: Dict[str, Any] = {
            "MethodName": node.name,
            "Description": docstring.strip(),
            "Parameters": self.extract_parameters(node),
            "Returns": self.extract_return_info(node),
        }
        if self.classes:  # Check if there are any classes to add methods to
            self.classes[-1]["Methods"].append(method_info)
            logging.info(
                f"Method {node.name} has been added to class {self.classes[-1]['ClassInformation']['ClassName']}."
            )
        else:
            logging.warning(
                f"Function {node.name} is defined outside of any class and will not be documented."
            )

    def extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, str]]:
        parameters: List[Dict[str, str]] = []
        for arg in node.args.args:
            try:
                param_type = self.infer_type(arg)
            except TypeInferenceError:
                param_type = "Type not specified"
            param_info: Dict[str, str] = {
                "Name": arg.arg,
                "Type": param_type,
                "Description": "Parameter description not provided.",
            }
            parameters.append(param_info)
        return parameters

    def extract_return_info(self, node: ast.FunctionDef) -> Dict[str, str]:
        return_annotation = node.returns
        return_type: str = (
            self.infer_type(return_annotation)
            if return_annotation
            else "Return type not specified"
        )
        return {
            "Type": return_type,
            "Description": "Detailed return information not available.",
        }


def parse_python_files(filepaths: List[str]) -> List[Dict[str, Any]]:
    classes: List[Dict[str, Any]] = []
    for filepath in filepaths:
        logging.info(f"Attempting to parse file: {filepath}")
        try:
            with open(filepath, "r") as file:
                node = ast.parse(file.read(), filename=os.path.basename(filepath))
                parser = CodeParser()
                parser.visit(node)
                classes.extend(parser.classes)
                logging.info(f"Successfully parsed {filepath}.")
        except SyntaxError as e:
            messagebox.showerror(
                "Error", f"Syntax error in {os.path.basename(filepath)}: {e}"
            )
            logging.error(f"Syntax error in {filepath}: {e}")
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to parse {os.path.basename(filepath)}: {e}"
            )
            logging.error(f"Failed to parse {filepath}: {e}")
    return classes


def save_json(data: List[Dict[str, Any]], path: str):
    output_file = os.path.join(path, "documentation.json")
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)
    messagebox.showinfo("Success", f"Documentation generated at {output_file}")
    logging.info(f"Documentation saved at {output_file}")


def browse_files() -> List[str]:
    filenames = filedialog.askopenfilenames(
        initialdir="/",
        title="Select Python Files",
        filetypes=(("Python files", "*.py*"), ("All files", "*.*")),
    )
    logging.debug(f"Files selected: {filenames}")
    return list(filenames)


def save_directory() -> str:
    directory = filedialog.askdirectory()
    logging.debug(f"Save directory chosen: {directory}")
    return directory


def generate_documentation():
    src_files = browse_files()
    if src_files:
        data = parse_python_files(src_files)
        if data:
            output_dir = save_directory()
            if output_dir:
                save_json(data, output_dir)
            else:
                messagebox.showerror("Error", "No save directory chosen.")
                logging.warning("No save directory chosen.")
    else:
        messagebox.showerror("Error", "No files chosen.")
        logging.warning("No files chosen.")


app = tk.Tk()
app.title("Python Class Documentation Generator")

browse_button = tk.Button(
    app, text="Browse Python Files", command=generate_documentation
)
browse_button.pack()

app.mainloop()
