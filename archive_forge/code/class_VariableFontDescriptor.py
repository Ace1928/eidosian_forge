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
class VariableFontDescriptor(SimpleDescriptor):
    """Container for variable fonts, sub-spaces of the Designspace.

    Use-cases:

    - From a single DesignSpace with discrete axes, define 1 variable font
      per value on the discrete axes. Before version 5, you would have needed
      1 DesignSpace per such variable font, and a lot of data duplication.
    - From a big variable font with many axes, define subsets of that variable
      font that only include some axes and freeze other axes at a given location.

    .. versionadded:: 5.0
    """
    flavor = 'variable-font'
    _attrs = ('filename', 'axisSubsets', 'lib')
    filename = posixpath_property('_filename')

    def __init__(self, *, name, filename=None, axisSubsets=None, lib=None):
        self.name: str = name
        'string, required. Name of this variable to identify it during the\n        build process and from other parts of the document, and also as a\n        filename in case the filename property is empty.\n\n        VarLib.\n        '
        self.filename: str = filename
        'string, optional. Relative path to the variable font file, **as it is\n        in the document**. The file may or may not exist.\n\n        If not specified, the :attr:`name` will be used as a basename for the file.\n        '
        self.axisSubsets: List[Union[RangeAxisSubsetDescriptor, ValueAxisSubsetDescriptor]] = axisSubsets or []
        'Axis subsets to include in this variable font.\n\n        If an axis is not mentioned, assume that we only want the default\n        location of that axis (same as a :class:`ValueAxisSubsetDescriptor`).\n        '
        self.lib: MutableMapping[str, Any] = lib or {}
        'Custom data associated with this variable font.'