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
def _getEffectiveFormatTuple(self):
    """Try to use the version specified in the document, or a sufficiently
        recent version to be able to encode what the document contains.
        """
    minVersion = self.documentObject.formatTuple
    if any((hasattr(axis, 'values') or axis.axisOrdering is not None or axis.axisLabels for axis in self.documentObject.axes)) or self.documentObject.locationLabels or any((source.localisedFamilyName for source in self.documentObject.sources)) or self.documentObject.variableFonts or any((instance.locationLabel or instance.userLocation for instance in self.documentObject.instances)):
        if minVersion < (5, 0):
            minVersion = (5, 0)
    if self.documentObject.axisMappings:
        if minVersion < (5, 1):
            minVersion = (5, 1)
    return minVersion