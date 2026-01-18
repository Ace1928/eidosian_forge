from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import httpx
import huggingface_hub
import websockets
from packaging import version
from gradio_client import serializing, utils
from gradio_client.exceptions import SerializationSetupError
from gradio_client.utils import (
def _add_uploaded_files_to_data(self, files: list[str | list[str]] | list[dict[str, Any] | list[dict[str, Any]]], data: list[Any]) -> None:
    """Helper function to modify the input data with the uploaded files."""
    file_counter = 0
    for i, t in enumerate(self.input_component_types):
        if t in ['file', 'uploadbutton']:
            data[i] = files[file_counter]
            file_counter += 1