import json
from pathlib import Path
from typing import Any, List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class NotebookLoader(BaseLoader):
    """Load `Jupyter notebook` (.ipynb) files."""

    def __init__(self, path: Union[str, Path], include_outputs: bool=False, max_output_length: int=10, remove_newline: bool=False, traceback: bool=False):
        """Initialize with a path.

        Args:
            path: The path to load the notebook from.
            include_outputs: Whether to include the outputs of the cell.
                Defaults to False.
            max_output_length: Maximum length of the output to be displayed.
                Defaults to 10.
            remove_newline: Whether to remove newlines from the notebook.
                Defaults to False.
            traceback: Whether to return a traceback of the error.
                Defaults to False.
        """
        self.file_path = path
        self.include_outputs = include_outputs
        self.max_output_length = max_output_length
        self.remove_newline = remove_newline
        self.traceback = traceback

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.file_path)
        with open(p, encoding='utf8') as f:
            d = json.load(f)
        filtered_data = [{k: v for k, v in cell.items() if k in ['cell_type', 'source', 'outputs']} for cell in d['cells']]
        if self.remove_newline:
            filtered_data = list(map(remove_newlines, filtered_data))
        text = ''.join(list(map(lambda x: concatenate_cells(x, self.include_outputs, self.max_output_length, self.traceback), filtered_data)))
        metadata = {'source': str(p)}
        return [Document(page_content=text, metadata=metadata)]