from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
def _addLib(self, parentElement: ET.Element, data: Any, indent_level: int) -> None:
    if not data:
        return
    libElement = ET.Element('lib')
    libElement.append(plistlib.totree(data, indent_level=indent_level))
    parentElement.append(libElement)