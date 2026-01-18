import os
import sys
import time
import traceback
import param
from IPython.display import Javascript, display
from nbconvert import HTMLExporter, NotebookExporter
from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
from nbformat import reader
from ..core.io import FileArchive, Pickler
from ..plotting.renderer import HTML_TAGS, MIME_TYPES
from .preprocessors import Substitute
def _clear_notebook(self, node):
    exporter = NotebookExporter()
    exporter.register_preprocessor(ClearOutputPreprocessor(enabled=True))
    cleared, _ = exporter.from_notebook_node(node)
    return cleared