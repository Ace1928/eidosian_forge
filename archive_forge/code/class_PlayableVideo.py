from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Iterable, Literal
import numpy as np
import PIL.Image
from gradio import components
from gradio.components.audio import WaveformOptions
from gradio.components.image_editor import Brush, Eraser
class PlayableVideo(components.Video):
    """
    Sets: format="mp4"
    """
    is_template = True

    def __init__(self, value: str | Path | tuple[str | Path, str | Path | None] | Callable | None=None, *, format: Literal['mp4']='mp4', sources: list[Literal['upload', 'webcam']] | None=None, height: int | str | None=None, width: int | str | None=None, label: str | None=None, every: float | None=None, show_label: bool | None=None, container: bool=True, scale: int | None=None, min_width: int=160, interactive: bool | None=None, visible: bool=True, elem_id: str | None=None, elem_classes: list[str] | str | None=None, render: bool=True, mirror_webcam: bool=True, include_audio: bool | None=None, autoplay: bool=False, show_share_button: bool | None=None, show_download_button: bool | None=None, min_length: int | None=None, max_length: int | None=None):
        sources = ['upload']
        super().__init__(value=value, format=format, sources=sources, height=height, width=width, label=label, every=every, show_label=show_label, container=container, scale=scale, min_width=min_width, interactive=interactive, visible=visible, elem_id=elem_id, elem_classes=elem_classes, render=render, mirror_webcam=mirror_webcam, include_audio=include_audio, autoplay=autoplay, show_share_button=show_share_button, show_download_button=show_download_button, min_length=min_length, max_length=max_length)