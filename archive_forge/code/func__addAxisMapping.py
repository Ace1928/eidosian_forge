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
def _addAxisMapping(self, mappingsElement, mappingObject):
    mappingElement = ET.Element('mapping')
    if getattr(mappingObject, 'description', None) is not None:
        mappingElement.attrib['description'] = mappingObject.description
    for what in ('inputLocation', 'outputLocation'):
        whatObject = getattr(mappingObject, what, None)
        if whatObject is None:
            continue
        whatElement = ET.Element(what[:-8])
        mappingElement.append(whatElement)
        for name, value in whatObject.items():
            dimensionElement = ET.Element('dimension')
            dimensionElement.attrib['name'] = name
            dimensionElement.attrib['xvalue'] = self.intOrFloat(value)
            whatElement.append(dimensionElement)
    mappingsElement.append(mappingElement)