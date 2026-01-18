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
def _readConditionElements(self, parentElement, ruleName=None):
    cds = []
    for conditionElement in parentElement.findall('.condition'):
        cd = {}
        cdMin = conditionElement.attrib.get('minimum')
        if cdMin is not None:
            cd['minimum'] = float(cdMin)
        else:
            cd['minimum'] = None
        cdMax = conditionElement.attrib.get('maximum')
        if cdMax is not None:
            cd['maximum'] = float(cdMax)
        else:
            cd['maximum'] = None
        cd['name'] = conditionElement.attrib.get('name')
        if cd.get('minimum') is None and cd.get('maximum') is None:
            raise DesignSpaceDocumentError('condition missing required minimum or maximum in rule' + (" '%s'" % ruleName if ruleName is not None else ''))
        cds.append(cd)
    return cds