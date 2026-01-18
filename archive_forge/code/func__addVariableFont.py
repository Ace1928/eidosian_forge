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
def _addVariableFont(self, parentElement: ET.Element, vf: VariableFontDescriptor) -> None:
    vfElement = ET.Element('variable-font')
    vfElement.attrib['name'] = vf.name
    if vf.filename is not None:
        vfElement.attrib['filename'] = vf.filename
    if vf.axisSubsets:
        subsetsElement = ET.Element('axis-subsets')
        for subset in vf.axisSubsets:
            subsetElement = ET.Element('axis-subset')
            subsetElement.attrib['name'] = subset.name
            if hasattr(subset, 'userMinimum'):
                subset = cast(RangeAxisSubsetDescriptor, subset)
                if subset.userMinimum != -math.inf:
                    subsetElement.attrib['userminimum'] = self.intOrFloat(subset.userMinimum)
                if subset.userMaximum != math.inf:
                    subsetElement.attrib['usermaximum'] = self.intOrFloat(subset.userMaximum)
                if subset.userDefault is not None:
                    subsetElement.attrib['userdefault'] = self.intOrFloat(subset.userDefault)
            elif hasattr(subset, 'userValue'):
                subset = cast(ValueAxisSubsetDescriptor, subset)
                subsetElement.attrib['uservalue'] = self.intOrFloat(subset.userValue)
            subsetsElement.append(subsetElement)
        vfElement.append(subsetsElement)
    self._addLib(vfElement, vf.lib, 4)
    parentElement.append(vfElement)