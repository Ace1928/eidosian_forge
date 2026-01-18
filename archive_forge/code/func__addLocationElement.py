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
def _addLocationElement(self, parentElement, *, designLocation: AnisotropicLocationDict=None, userLocation: SimpleLocationDict=None):
    locElement = ET.Element('location')
    for axis in self.documentObject.axes:
        if designLocation is not None and axis.name in designLocation:
            dimElement = ET.Element('dimension')
            dimElement.attrib['name'] = axis.name
            value = designLocation[axis.name]
            if isinstance(value, tuple):
                dimElement.attrib['xvalue'] = self.intOrFloat(value[0])
                dimElement.attrib['yvalue'] = self.intOrFloat(value[1])
            else:
                dimElement.attrib['xvalue'] = self.intOrFloat(value)
            locElement.append(dimElement)
        elif userLocation is not None and axis.name in userLocation:
            dimElement = ET.Element('dimension')
            dimElement.attrib['name'] = axis.name
            value = userLocation[axis.name]
            dimElement.attrib['uservalue'] = self.intOrFloat(value)
            locElement.append(dimElement)
    if len(locElement) > 0:
        parentElement.append(locElement)