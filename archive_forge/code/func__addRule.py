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
def _addRule(self, ruleObject):
    ruleElement = ET.Element('rule')
    if ruleObject.name is not None:
        ruleElement.attrib['name'] = ruleObject.name
    for conditions in ruleObject.conditionSets:
        conditionsetElement = ET.Element('conditionset')
        for cond in conditions:
            if cond.get('minimum') is None and cond.get('maximum') is None:
                continue
            conditionElement = ET.Element('condition')
            conditionElement.attrib['name'] = cond.get('name')
            if cond.get('minimum') is not None:
                conditionElement.attrib['minimum'] = self.intOrFloat(cond.get('minimum'))
            if cond.get('maximum') is not None:
                conditionElement.attrib['maximum'] = self.intOrFloat(cond.get('maximum'))
            conditionsetElement.append(conditionElement)
        if len(conditionsetElement):
            ruleElement.append(conditionsetElement)
    for sub in ruleObject.subs:
        subElement = ET.Element('sub')
        subElement.attrib['name'] = sub[0]
        subElement.attrib['with'] = sub[1]
        ruleElement.append(subElement)
    if len(ruleElement):
        self.root.findall('.rules')[0].append(ruleElement)