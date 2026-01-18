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
def _makeLocationElement(self, locationObject, name=None):
    """Convert Location dict to a locationElement."""
    locElement = ET.Element('location')
    if name is not None:
        locElement.attrib['name'] = name
    validatedLocation = self.documentObject.newDefaultLocation()
    for axisName, axisValue in locationObject.items():
        if axisName in validatedLocation:
            validatedLocation[axisName] = axisValue
    for dimensionName, dimensionValue in validatedLocation.items():
        dimElement = ET.Element('dimension')
        dimElement.attrib['name'] = dimensionName
        if type(dimensionValue) == tuple:
            dimElement.attrib['xvalue'] = self.intOrFloat(dimensionValue[0])
            dimElement.attrib['yvalue'] = self.intOrFloat(dimensionValue[1])
        else:
            dimElement.attrib['xvalue'] = self.intOrFloat(dimensionValue)
        locElement.append(dimElement)
    return (locElement, validatedLocation)