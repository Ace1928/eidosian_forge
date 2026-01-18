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
class Eraser:
    """
    A dataclass for specifying options for the eraser tool in the ImageEditor component. An instance of this class can be passed to the `eraser` parameter of `gr.ImageEditor`.
    Parameters:
        default_size: The default radius, in pixels, of the eraser tool. Defaults to "auto" in which case the radius is automatically determined based on the size of the image (generally 1/50th of smaller dimension).
    """
    default_size: int | Literal['auto'] = 'auto'