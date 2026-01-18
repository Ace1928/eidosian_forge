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
class AsDictMixin(object):

    def asdict(self):
        d = {}
        for attr, value in self.__dict__.items():
            if attr.startswith('_'):
                continue
            if hasattr(value, 'asdict'):
                value = value.asdict()
            elif isinstance(value, list):
                value = [v.asdict() if hasattr(v, 'asdict') else v for v in value]
            d[attr] = value
        return d