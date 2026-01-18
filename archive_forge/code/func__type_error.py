from __future__ import annotations
import asyncio
import base64
import struct
from io import BytesIO
from pathlib import PurePath
from typing import (
import param
from ..models import PDF as _BkPDF
from ..util import isfile, isurl
from .markup import HTMLBasePane, escape
def _type_error(self, object):
    if isinstance(object, str):
        raise ValueError('%s pane cannot parse string that is not a filename, URL or a SVG XML contents.' % type(self).__name__)
    super()._type_error(object)