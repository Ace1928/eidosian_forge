from __future__ import annotations
import io
import os
import re
from enum import IntEnum
from typing import TYPE_CHECKING, Final, List, Literal, Sequence, Union, cast
from typing_extensions import TypeAlias
from streamlit import runtime, url_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Image_pb2 import ImageList as ImageListProto
from streamlit.runtime import caching
from streamlit.runtime.metrics_util import gather_metrics
def _validate_image_format_string(image_data: bytes | PILImage, format: str) -> ImageFormat:
    """Return either "JPEG", "PNG", or "GIF", based on the input `format` string.

    - If `format` is "JPEG" or "JPG" (or any capitalization thereof), return "JPEG"
    - If `format` is "PNG" (or any capitalization thereof), return "PNG"
    - For all other strings, return "PNG" if the image has an alpha channel,
    "GIF" if the image is a GIF, and "JPEG" otherwise.
    """
    format = format.upper()
    if format == 'JPEG' or format == 'PNG':
        return cast(ImageFormat, format)
    if format == 'JPG':
        return 'JPEG'
    if isinstance(image_data, bytes):
        from PIL import Image
        pil_image = Image.open(io.BytesIO(image_data))
    else:
        pil_image = image_data
    if _image_is_gif(pil_image):
        return 'GIF'
    if _image_may_have_alpha_channel(pil_image):
        return 'PNG'
    return 'JPEG'