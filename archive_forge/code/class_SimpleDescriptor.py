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
class SimpleDescriptor(AsDictMixin):
    """Containers for a bunch of attributes"""

    def compare(self, other):
        for attr in self._attrs:
            try:
                assert getattr(self, attr) == getattr(other, attr)
            except AssertionError:
                print('failed attribute', attr, getattr(self, attr), '!=', getattr(other, attr))

    def __repr__(self):
        attrs = [f'{a}={repr(getattr(self, a))},' for a in self._attrs]
        attrs = indent('\n'.join(attrs), '    ')
        return f'{self.__class__.__name__}(\n{attrs}\n)'