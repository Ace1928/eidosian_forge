from __future__ import annotations
import dataclasses
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
import numpy as np
import PIL.Image
from gradio_client import file
from gradio_client.documentation import document
from typing_extensions import TypedDict
from gradio import image_utils, utils
from gradio.components.base import Component, server
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events
@dataclasses.dataclass
class Brush(Eraser):
    """
    A dataclass for specifying options for the brush tool in the ImageEditor component. An instance of this class can be passed to the `brush` parameter of `gr.ImageEditor`.
    Parameters:
        default_size: The default radius, in pixels, of the brush tool. Defaults to "auto" in which case the radius is automatically determined based on the size of the image (generally 1/50th of smaller dimension).
        colors: A list of colors to make available to the user when using the brush. Defaults to a list of 5 colors.
        default_color: The default color of the brush. Defaults to the first color in the `colors` list.
        color_mode: If set to "fixed", user can only select from among the colors in `colors`. If "defaults", the colors in `colors` are provided as a default palette, but the user can also select any color using a color picker.
    """
    colors: Union[list[str], str, None] = None
    default_color: Union[str, Literal['auto']] = 'auto'
    color_mode: Literal['fixed', 'defaults'] = 'defaults'

    def __post_init__(self):
        if self.colors is None:
            self.colors = ['rgb(204, 50, 50)', 'rgb(173, 204, 50)', 'rgb(50, 204, 112)', 'rgb(50, 112, 204)', 'rgb(173, 50, 204)']
        if self.default_color is None:
            self.default_color = self.colors[0] if isinstance(self.colors, list) else self.colors