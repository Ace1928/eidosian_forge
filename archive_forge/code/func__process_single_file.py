from __future__ import annotations
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Literal
import gradio_client.utils as client_utils
from gradio_client import file
from gradio_client.documentation import document
from gradio import processing_utils
from gradio.components.base import Component
from gradio.data_classes import FileData, ListFiles
from gradio.events import Events
from gradio.utils import NamedString
def _process_single_file(self, f: FileData) -> bytes | NamedString:
    file_name = f.path
    if self.type == 'filepath':
        file = tempfile.NamedTemporaryFile(delete=False, dir=self.GRADIO_CACHE)
        file.name = file_name
        return NamedString(file_name)
    elif self.type == 'binary':
        with open(file_name, 'rb') as file_data:
            return file_data.read()
    else:
        raise ValueError('Unknown type: ' + str(type) + ". Please choose from: 'filepath', 'binary'.")