from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Iterable, Literal
import numpy as np
import PIL.Image
from gradio import components
from gradio.components.audio import WaveformOptions
from gradio.components.image_editor import Brush, Eraser
class Microphone(components.Audio):
    """
    Sets: sources=["microphone"]
    """
    is_template = True

    def __init__(self, value: str | Path | tuple[int, np.ndarray] | Callable | None=None, *, sources: list[Literal['upload', 'microphone']] | None=None, type: Literal['numpy', 'filepath']='numpy', label: str | None=None, every: float | None=None, show_label: bool | None=None, container: bool=True, scale: int | None=None, min_width: int=160, interactive: bool | None=None, visible: bool=True, streaming: bool=False, elem_id: str | None=None, elem_classes: list[str] | str | None=None, render: bool=True, format: Literal['wav', 'mp3']='wav', autoplay: bool=False, show_download_button: bool | None=None, show_share_button: bool | None=None, editable: bool=True, min_length: int | None=None, max_length: int | None=None, waveform_options: WaveformOptions | dict | None=None):
        sources = ['microphone']
        super().__init__(value, sources=sources, type=type, label=label, every=every, show_label=show_label, container=container, scale=scale, min_width=min_width, interactive=interactive, visible=visible, streaming=streaming, elem_id=elem_id, elem_classes=elem_classes, render=render, format=format, autoplay=autoplay, show_download_button=show_download_button, show_share_button=show_share_button, editable=editable, min_length=min_length, max_length=max_length, waveform_options=waveform_options)