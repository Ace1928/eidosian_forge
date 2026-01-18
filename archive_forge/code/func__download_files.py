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
def _download_files(self, value: str | list[str]) -> str | list[str]:
    downloaded_files = []
    if isinstance(value, list):
        for file in value:
            if client_utils.is_http_url_like(file):
                downloaded_file = processing_utils.save_url_to_cache(file, self.GRADIO_CACHE)
                downloaded_files.append(downloaded_file)
            else:
                downloaded_files.append(file)
        return downloaded_files
    if client_utils.is_http_url_like(value):
        downloaded_file = processing_utils.save_url_to_cache(value, self.GRADIO_CACHE)
        return downloaded_file
    else:
        return value