from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Iterable, Literal
import numpy as np
import PIL.Image
from gradio import components
from gradio.components.audio import WaveformOptions
from gradio.components.image_editor import Brush, Eraser
class Sketchpad(components.ImageEditor):
    """
    Sets: sources=(), brush=Brush(colors=["#000000"], color_mode="fixed")
    """
    is_template = True

    def __init__(self, value: str | PIL.Image.Image | np.ndarray | None=None, *, height: int | str | None=None, width: int | str | None=None, image_mode: Literal['1', 'L', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV', 'I', 'F']='RGBA', sources: Iterable[Literal['upload', 'webcam', 'clipboard']]=(), type: Literal['numpy', 'pil', 'filepath']='numpy', label: str | None=None, every: float | None=None, show_label: bool | None=None, show_download_button: bool=True, container: bool=True, scale: int | None=None, min_width: int=160, interactive: bool | None=None, visible: bool=True, elem_id: str | None=None, elem_classes: list[str] | str | None=None, render: bool=True, mirror_webcam: bool=True, show_share_button: bool | None=None, _selectable: bool=False, crop_size: tuple[int | float, int | float] | str | None=None, transforms: Iterable[Literal['crop']]=('crop',), eraser: Eraser | None=None, brush: Brush | None=None, format: str='webp', layers: bool=True):
        if not brush:
            brush = Brush(colors=['#000000'], color_mode='fixed')
        super().__init__(value=value, height=height, width=width, image_mode=image_mode, sources=sources, type=type, label=label, every=every, show_label=show_label, show_download_button=show_download_button, container=container, scale=scale, min_width=min_width, interactive=interactive, visible=visible, elem_id=elem_id, elem_classes=elem_classes, render=render, mirror_webcam=mirror_webcam, show_share_button=show_share_button, _selectable=_selectable, crop_size=crop_size, transforms=transforms, eraser=eraser, brush=brush, format=format, layers=layers)