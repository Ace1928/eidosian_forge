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
def _addAxis(self, axisObject):
    axisElement = ET.Element('axis')
    axisElement.attrib['tag'] = axisObject.tag
    axisElement.attrib['name'] = axisObject.name
    self._addLabelNames(axisElement, axisObject.labelNames)
    if axisObject.map:
        for inputValue, outputValue in axisObject.map:
            mapElement = ET.Element('map')
            mapElement.attrib['input'] = self.intOrFloat(inputValue)
            mapElement.attrib['output'] = self.intOrFloat(outputValue)
            axisElement.append(mapElement)
    if axisObject.axisOrdering or axisObject.axisLabels:
        labelsElement = ET.Element('labels')
        if axisObject.axisOrdering is not None:
            labelsElement.attrib['ordering'] = str(axisObject.axisOrdering)
        for label in axisObject.axisLabels:
            self._addAxisLabel(labelsElement, label)
        axisElement.append(labelsElement)
    if hasattr(axisObject, 'minimum'):
        axisElement.attrib['minimum'] = self.intOrFloat(axisObject.minimum)
        axisElement.attrib['maximum'] = self.intOrFloat(axisObject.maximum)
    elif hasattr(axisObject, 'values'):
        axisElement.attrib['values'] = ' '.join((self.intOrFloat(v) for v in axisObject.values))
    axisElement.attrib['default'] = self.intOrFloat(axisObject.default)
    if axisObject.hidden:
        axisElement.attrib['hidden'] = '1'
    self.root.findall('.axes')[0].append(axisElement)