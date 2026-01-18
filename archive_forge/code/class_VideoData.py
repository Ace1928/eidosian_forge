from __future__ import annotations
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Optional
from gradio_client import file
from gradio_client import utils as client_utils
from gradio_client.documentation import document
import gradio as gr
from gradio import processing_utils, utils, wasm_utils
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events
class VideoData(GradioModel):
    video: FileData
    subtitles: Optional[FileData] = None