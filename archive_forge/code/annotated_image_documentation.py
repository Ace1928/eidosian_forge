from __future__ import annotations
from typing import Any, List
import gradio_client.utils as client_utils
import numpy as np
import PIL.Image
from gradio_client import file
from gradio_client.documentation import document
from gradio import processing_utils, utils
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events

        Parameters:
            value: Expects a a tuple of a base image and list of annotations: a `tuple[Image, list[Annotation]]`. The `Image` itself can be `str` filepath, `numpy.ndarray`, or `PIL.Image`. Each `Annotation` is a `tuple[Mask, str]`. The `Mask` can be either a `tuple` of 4 `int`'s representing the bounding box coordinates (x1, y1, x2, y2), or 0-1 confidence mask in the form of a `numpy.ndarray` of the same shape as the image, while the second element of the `Annotation` tuple is a `str` label.
        Returns:
            Tuple of base image file and list of annotations, with each annotation a two-part tuple where the first element image path of the mask, and the second element is the label.
        