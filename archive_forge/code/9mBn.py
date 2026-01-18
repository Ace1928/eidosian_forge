import ast
import json
import tkinter as tk
from tkinter import filedialog, messagebox, Tk, Button
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

import docstring_parser


# Configure detailed logging with advanced settings
def setup_logging() -> None:
    """
    Sets up advanced logging with both console and file handlers to ensure comprehensive logging coverage.
    This function configures the logging to capture a wide range of information about the program's operation,
    which is crucial for debugging and monitoring the application's behavior.
    """
    logger: logging.Logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers for both console and file output
    c_handler: logging.StreamHandler = logging.StreamHandler()
    f_handler: logging.FileHandler = logging.FileHandler("docparser.log", mode="w")
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    c_format: logging.Formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s"
    )
    f_format: logging.Formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


# Initialize logging as soon as possible to capture all events
setup_logging()


class TypeInferenceError(Exception):
    """
    Custom exception for type inference errors, providing a clear, detailed, and structured description of the error context.
    This exception is raised when the type inference process encounters an unresolvable situation, ensuring that such
    critical issues are not silently ignored but are logged and handled appropriately.

    Attributes:
        message (str): Explanation of the error, detailing what was attempted and why it failed.
    """

    def __init__(self, message: str) -> None:
        """
        Initializes the TypeInferenceError with an error message and logs the error.

        Parameters:
            message (str): The message describing the type inference error.
        """
        self.message: str = message
        super().__init__(message)
        logging.error(f"TypeInferenceError: {message}")

    def __str__(self) -> str:
        """
        Returns a string representation of the error, enhancing error traceability by providing a clear and detailed message.

        Returns:
            str: A string that represents the error message, which includes the context of the error.
        """
        return f"TypeInferenceError occurred with message: {self.message}"


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

    def extract_properties(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """
        Extracts properties from a class definition using AST nodes.

        Parameters:
            node (ast.ClassDef): The class definition from which properties are extracted.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a property with its name, type, and description.
        """
        properties: List[Dict[str, Any]] = []
        for assign in node.body:
            if isinstance(assign, ast.AnnAssign):
                property_info: Dict[str, Any] = {
                    "Name": assign.target.id,
                    "Type": (
                        self.infer_type(assign.annotation)
                        if assign.annotation
                        else "Type not determined"
                    ),
                    "Description": ast.get_docstring(assign)
                    or "No property description available.",
                }
                properties.append(property_info)
                logging.debug(f"Property extracted: {property_info}")
        logging.debug(f"Properties extracted from {node.name}: {properties}")
        return properties

    def parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """
        Parses a docstring to extract detailed information.

        Parameters:
            docstring (str): The docstring to parse.

        Returns:
            Dict[str, Any]: A dictionary containing parsed docstring components such as short description, long description, parameters, returns, raises, examples, and see also references.
        """
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

    def enhance_method_info(self, method_info: Dict[str, Any], docstring: str) -> None:
        """
        Enhances the method information dictionary with parsed docstring details.

        Parameters:
            method_info (Dict[str, Any]): The dictionary containing method information to be enhanced.
            docstring (str): The docstring from which additional information is extracted.

        Returns:
            None
        """
        enhanced_doc = self.parse_docstring(docstring)
        method_info.update(enhanced_doc)
        logging.debug(f"Method information enhanced with docstring: {enhanced_doc}")

    def infer_type(self, assign: Union[ast.AnnAssign, ast.arg]) -> str:
        """
        Infers the type of a given assignment or argument node in the abstract syntax tree (AST).
        This method has been enhanced to handle more complex type annotations including generics and union types,
        which are common in modern Python codebases. It is robust, handling nested types and providing detailed error logging.

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

    def _infer_complex_type(self, annotation) -> str:
        """
        Recursively handles complex type annotations like Union, List, Dict, etc., by parsing the AST nodes.
        This method supports advanced type structures, enhancing the capability of the type inference system.

        Parameters:
            annotation (ast.expr): The annotation node from AST representing the type.

        Returns:
            str: The inferred type as a string, formatted appropriately for complex types.
        """
        if isinstance(annotation, ast.Subscript):
            base = self._infer_complex_type(
                annotation.value
            )  # Recursively infer the base type
            if isinstance(annotation.slice, ast.Index):  # Python 3.8 and earlier
                index = self._infer_complex_type(annotation.slice.value)
            else:  # Python 3.9+
                index = ", ".join(
                    self._infer_complex_type(s.value) for s in annotation.slice.values
                )
            return f"{base}[{index}]"
        elif isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Tuple):
            elements = ", ".join(self._infer_complex_type(el) for el in annotation.elts)
            return f"Tuple[{elements}]"
        elif isinstance(annotation, ast.Attribute):
            return f"{self._infer_complex_type(annotation.value)}.{annotation.attr}"
        return "Unknown"

    def _infer_arg_type(self, assign: ast.arg) -> str:
        """
        Comprehensive method to infer type from an arg node, enhanced to handle complex types using the _infer_complex_type method.
        This method supports a broader range of type annotations, improving the accuracy and depth of type inference.

        Parameters:
            assign (ast.arg): The arg node from AST.

        Returns:
            str: The inferred type as a string, handling complex annotations including Names, Attributes, Subscripts, and ExtSlices.
        """
        if assign.annotation:
            # Directly handle simple Name annotations
            if isinstance(assign.annotation, ast.Name):
                return assign.annotation.id
            # Handle complex annotations by delegating to the _infer_complex_type method
            elif isinstance(assign.annotation, (ast.Attribute, ast.Subscript)):
                return self._infer_complex_type(assign.annotation)
            else:
                # Log the occurrence of an unexpected annotation type
                logging.warning(
                    f"Encountered an unexpected annotation type: {type(assign.annotation).__name__}"
                )
        # Return 'Unknown' if no annotation is present or if the annotation type is not handled
        return "Unknown"


class CodeParser(ast.NodeVisitor):
    """
    A class to parse Python code and extract class and function definitions with detailed documentation.
    """

    def __init__(self):
        self.classes: List[Dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visit a class definition and extract relevant information including the name, docstring, and methods.

        Parameters:
            node (ast.ClassDef): The class definition node in the AST.
        """
        class_info: Dict[str, Any] = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "methods": [],
        }
        self.generic_visit(node)
        self.classes.append(class_info)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit a function definition and extract relevant information including the name, docstring, and parameters.

        Parameters:
            node (ast.FunctionDef): The function definition node in the AST.
        """
        func_info: Dict[str, Any] = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "parameters": [arg.arg for arg in node.args.args],
        }
        self.classes[-1]["methods"].append(func_info)
        self.generic_visit(node)

    def _infer_complex_type(self, annotation: ast.AST) -> str:
        """
        Recursively handles complex type annotations like Union, List, Dict, etc., and returns a string representation.

        Parameters:
            annotation (ast.AST): The AST node representing the type annotation.

        Returns:
            str: A string representation of the type annotation.
        """
        if isinstance(annotation, ast.Subscript):
            base: str = self._infer_complex_type(annotation.value)
            if isinstance(annotation.slice, ast.Index):  # Python 3.8 and earlier
                index: str = self._infer_complex_type(annotation.slice.value)
            else:  # Python 3.9+
                index: str = ", ".join(
                    self._infer_complex_type(s.value) for s in annotation.slice.values
                )
            return f"{base}[{index}]"
        elif isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Tuple):
            elements: str = ", ".join(
                self._infer_complex_type(el) for el in annotation.elts
            )
            return f"Tuple[{elements}]"
        elif isinstance(annotation, ast.Attribute):
            return f"{self._infer_complex_type(annotation.value)}.{annotation.attr}"
        return "Unknown"

    def _infer_arg_type(self, assign: ast.arg) -> str:
        """
        Enhanced to handle complex types using the _infer_complex_type method.

        Parameters:
            assign (ast.arg): The argument node from AST.

        Returns:
            str: The inferred type as a string, handling complex annotations.
        """
        if assign.annotation:
            return self._infer_complex_type(assign.annotation)
        return "Unknown"

    def parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """
        Parses docstrings to extract detailed information including descriptions, parameters, returns, raises, examples, and references.

        Parameters:
            docstring (str): The docstring to parse.

        Returns:
            Dict[str, Any]: A dictionary containing detailed information extracted from the docstring.
        """
        parsed: docstring_parser.Docstring = docstring_parser.parse(docstring)
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


def parse_python_files(filepaths: List[str]) -> List[Dict[str, Any]]:
    """
    Enhanced error handling and logging for parsing Python files.

    Parameters:
        filepaths (List[str]): A list of file paths to parse.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing class information from parsed files.
    """
    classes: List[Dict[str, Any]] = []
    for filepath in filepaths:
        logging.info(f"Attempting to parse file: {filepath}")
        try:
            with open(filepath, "r") as file:
                content: str = file.read()
                node: ast.AST = ast.parse(content, filename=os.path.basename(filepath))
                parser: CodeParser = CodeParser()
                parser.visit(node)
                classes.extend(parser.classes)
                logging.info(f"Successfully parsed {filepath}.")
        except SyntaxError as e:
            logging.error(f"Syntax error in {filepath}: {e}")
            continue  # Skip files with syntax errors
        except Exception as e:
            logging.error(f"Failed to parse {filepath}: {e}")
            continue
    return classes


def save_json(data: List[Dict[str, Any]], path: str) -> None:
    """
    Saves the parsed data into a JSON file.

    Parameters:
        data (List[Dict[str, Any]]): The data to save.
        path (str): The path where the JSON file will be saved.
    """
    output_file: str = os.path.join(path, "documentation.json")
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)
    messagebox.showinfo("Success", f"Documentation generated at {output_file}")
    logging.info(f"Documentation saved at {output_file}")


def browse_files() -> List[str]:
    """
    Opens a file dialog to select Python files for documentation.

    Returns:
        List[str]: A list of selected file names.
    """
    filenames: Tuple[str, ...] = filedialog.askopenfilenames(
        initialdir="/",
        title="Select Python Files",
        filetypes=(("Python files", "*.py*"), ("All files", "*.*")),
    )
    logging.debug(f"Files selected: {filenames}")
    return list(filenames)


def save_directory() -> str:
    """
    Opens a directory dialog to select the save location for the documentation.

    Returns:
        str: The selected directory path.
    """
    directory: str = filedialog.askdirectory()
    logging.debug(f"Save directory chosen: {directory}")
    return directory


def generate_documentation() -> None:
    """
    Orchestrates the documentation generation process.
    """
    src_files: List[str] = browse_files()
    if src_files:
        data: List[Dict[str, Any]] = parse_python_files(src_files)
        if data:
            output_dir: str = save_directory()
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
