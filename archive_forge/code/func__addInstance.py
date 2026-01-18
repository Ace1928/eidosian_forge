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
def _addInstance(self, instanceObject):
    instanceElement = ET.Element('instance')
    if instanceObject.name is not None:
        instanceElement.attrib['name'] = instanceObject.name
    if instanceObject.locationLabel is not None:
        instanceElement.attrib['location'] = instanceObject.locationLabel
    if instanceObject.familyName is not None:
        instanceElement.attrib['familyname'] = instanceObject.familyName
    if instanceObject.styleName is not None:
        instanceElement.attrib['stylename'] = instanceObject.styleName
    if instanceObject.localisedStyleName:
        languageCodes = list(instanceObject.localisedStyleName.keys())
        languageCodes.sort()
        for code in languageCodes:
            if code == 'en':
                continue
            localisedStyleNameElement = ET.Element('stylename')
            localisedStyleNameElement.attrib[XML_LANG] = code
            localisedStyleNameElement.text = instanceObject.getStyleName(code)
            instanceElement.append(localisedStyleNameElement)
    if instanceObject.localisedFamilyName:
        languageCodes = list(instanceObject.localisedFamilyName.keys())
        languageCodes.sort()
        for code in languageCodes:
            if code == 'en':
                continue
            localisedFamilyNameElement = ET.Element('familyname')
            localisedFamilyNameElement.attrib[XML_LANG] = code
            localisedFamilyNameElement.text = instanceObject.getFamilyName(code)
            instanceElement.append(localisedFamilyNameElement)
    if instanceObject.localisedStyleMapStyleName:
        languageCodes = list(instanceObject.localisedStyleMapStyleName.keys())
        languageCodes.sort()
        for code in languageCodes:
            if code == 'en':
                continue
            localisedStyleMapStyleNameElement = ET.Element('stylemapstylename')
            localisedStyleMapStyleNameElement.attrib[XML_LANG] = code
            localisedStyleMapStyleNameElement.text = instanceObject.getStyleMapStyleName(code)
            instanceElement.append(localisedStyleMapStyleNameElement)
    if instanceObject.localisedStyleMapFamilyName:
        languageCodes = list(instanceObject.localisedStyleMapFamilyName.keys())
        languageCodes.sort()
        for code in languageCodes:
            if code == 'en':
                continue
            localisedStyleMapFamilyNameElement = ET.Element('stylemapfamilyname')
            localisedStyleMapFamilyNameElement.attrib[XML_LANG] = code
            localisedStyleMapFamilyNameElement.text = instanceObject.getStyleMapFamilyName(code)
            instanceElement.append(localisedStyleMapFamilyNameElement)
    if self.effectiveFormatTuple >= (5, 0):
        if instanceObject.locationLabel is None:
            self._addLocationElement(instanceElement, designLocation=instanceObject.designLocation, userLocation=instanceObject.userLocation)
    elif instanceObject.location is not None:
        locationElement, instanceObject.location = self._makeLocationElement(instanceObject.location)
        instanceElement.append(locationElement)
    if instanceObject.filename is not None:
        instanceElement.attrib['filename'] = instanceObject.filename
    if instanceObject.postScriptFontName is not None:
        instanceElement.attrib['postscriptfontname'] = instanceObject.postScriptFontName
    if instanceObject.styleMapFamilyName is not None:
        instanceElement.attrib['stylemapfamilyname'] = instanceObject.styleMapFamilyName
    if instanceObject.styleMapStyleName is not None:
        instanceElement.attrib['stylemapstylename'] = instanceObject.styleMapStyleName
    if self.effectiveFormatTuple < (5, 0):
        if instanceObject.glyphs:
            if instanceElement.findall('.glyphs') == []:
                glyphsElement = ET.Element('glyphs')
                instanceElement.append(glyphsElement)
            glyphsElement = instanceElement.findall('.glyphs')[0]
            for glyphName, data in sorted(instanceObject.glyphs.items()):
                glyphElement = self._writeGlyphElement(instanceElement, instanceObject, glyphName, data)
                glyphsElement.append(glyphElement)
        if instanceObject.kerning:
            kerningElement = ET.Element('kerning')
            instanceElement.append(kerningElement)
        if instanceObject.info:
            infoElement = ET.Element('info')
            instanceElement.append(infoElement)
    self._addLib(instanceElement, instanceObject.lib, 4)
    self.root.findall('.instances')[0].append(instanceElement)