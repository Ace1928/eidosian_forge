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
def _PIL_to_bytes(image: PILImage, format: ImageFormat='JPEG', quality: int=100) -> bytes:
    """Convert a PIL image to bytes."""
    tmp = io.BytesIO()
    if format == 'JPEG' and _image_may_have_alpha_channel(image):
        image = image.convert('RGB')
    image.save(tmp, format=format, quality=quality)
    return tmp.getvalue()