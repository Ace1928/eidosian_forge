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
def _addSource(self, sourceObject):
    sourceElement = ET.Element('source')
    if sourceObject.filename is not None:
        sourceElement.attrib['filename'] = sourceObject.filename
    if sourceObject.name is not None:
        if sourceObject.name.find('temp_master') != 0:
            sourceElement.attrib['name'] = sourceObject.name
    if sourceObject.familyName is not None:
        sourceElement.attrib['familyname'] = sourceObject.familyName
    if sourceObject.styleName is not None:
        sourceElement.attrib['stylename'] = sourceObject.styleName
    if sourceObject.layerName is not None:
        sourceElement.attrib['layer'] = sourceObject.layerName
    if sourceObject.localisedFamilyName:
        languageCodes = list(sourceObject.localisedFamilyName.keys())
        languageCodes.sort()
        for code in languageCodes:
            if code == 'en':
                continue
            localisedFamilyNameElement = ET.Element('familyname')
            localisedFamilyNameElement.attrib[XML_LANG] = code
            localisedFamilyNameElement.text = sourceObject.getFamilyName(code)
            sourceElement.append(localisedFamilyNameElement)
    if sourceObject.copyLib:
        libElement = ET.Element('lib')
        libElement.attrib['copy'] = '1'
        sourceElement.append(libElement)
    if sourceObject.copyGroups:
        groupsElement = ET.Element('groups')
        groupsElement.attrib['copy'] = '1'
        sourceElement.append(groupsElement)
    if sourceObject.copyFeatures:
        featuresElement = ET.Element('features')
        featuresElement.attrib['copy'] = '1'
        sourceElement.append(featuresElement)
    if sourceObject.copyInfo or sourceObject.muteInfo:
        infoElement = ET.Element('info')
        if sourceObject.copyInfo:
            infoElement.attrib['copy'] = '1'
        if sourceObject.muteInfo:
            infoElement.attrib['mute'] = '1'
        sourceElement.append(infoElement)
    if sourceObject.muteKerning:
        kerningElement = ET.Element('kerning')
        kerningElement.attrib['mute'] = '1'
        sourceElement.append(kerningElement)
    if sourceObject.mutedGlyphNames:
        for name in sourceObject.mutedGlyphNames:
            glyphElement = ET.Element('glyph')
            glyphElement.attrib['name'] = name
            glyphElement.attrib['mute'] = '1'
            sourceElement.append(glyphElement)
    if self.effectiveFormatTuple >= (5, 0):
        self._addLocationElement(sourceElement, designLocation=sourceObject.location)
    else:
        locationElement, sourceObject.location = self._makeLocationElement(sourceObject.location)
        sourceElement.append(locationElement)
    self.root.findall('.sources')[0].append(sourceElement)