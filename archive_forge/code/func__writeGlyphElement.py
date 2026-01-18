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
def _writeGlyphElement(self, instanceElement, instanceObject, glyphName, data):
    glyphElement = ET.Element('glyph')
    if data.get('mute'):
        glyphElement.attrib['mute'] = '1'
    if data.get('unicodes') is not None:
        glyphElement.attrib['unicode'] = ' '.join([hex(u) for u in data.get('unicodes')])
    if data.get('instanceLocation') is not None:
        locationElement, data['instanceLocation'] = self._makeLocationElement(data.get('instanceLocation'))
        glyphElement.append(locationElement)
    if glyphName is not None:
        glyphElement.attrib['name'] = glyphName
    if data.get('note') is not None:
        noteElement = ET.Element('note')
        noteElement.text = data.get('note')
        glyphElement.append(noteElement)
    if data.get('masters') is not None:
        mastersElement = ET.Element('masters')
        for m in data.get('masters'):
            masterElement = ET.Element('master')
            if m.get('glyphName') is not None:
                masterElement.attrib['glyphname'] = m.get('glyphName')
            if m.get('font') is not None:
                masterElement.attrib['source'] = m.get('font')
            if m.get('location') is not None:
                locationElement, m['location'] = self._makeLocationElement(m.get('location'))
                masterElement.append(locationElement)
            mastersElement.append(masterElement)
        glyphElement.append(mastersElement)
    return glyphElement