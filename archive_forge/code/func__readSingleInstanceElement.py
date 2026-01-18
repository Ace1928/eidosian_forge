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
def _readSingleInstanceElement(self, instanceElement, makeGlyphs=True, makeKerning=True, makeInfo=True):
    filename = instanceElement.attrib.get('filename')
    if filename is not None and self.documentObject.path is not None:
        instancePath = os.path.join(os.path.dirname(self.documentObject.path), filename)
    else:
        instancePath = None
    instanceObject = self.instanceDescriptorClass()
    instanceObject.path = instancePath
    instanceObject.filename = filename
    name = instanceElement.attrib.get('name')
    if name is not None:
        instanceObject.name = name
    familyname = instanceElement.attrib.get('familyname')
    if familyname is not None:
        instanceObject.familyName = familyname
    stylename = instanceElement.attrib.get('stylename')
    if stylename is not None:
        instanceObject.styleName = stylename
    postScriptFontName = instanceElement.attrib.get('postscriptfontname')
    if postScriptFontName is not None:
        instanceObject.postScriptFontName = postScriptFontName
    styleMapFamilyName = instanceElement.attrib.get('stylemapfamilyname')
    if styleMapFamilyName is not None:
        instanceObject.styleMapFamilyName = styleMapFamilyName
    styleMapStyleName = instanceElement.attrib.get('stylemapstylename')
    if styleMapStyleName is not None:
        instanceObject.styleMapStyleName = styleMapStyleName
    for styleNameElement in instanceElement.findall('stylename'):
        for key, lang in styleNameElement.items():
            if key == XML_LANG:
                styleName = styleNameElement.text
                instanceObject.setStyleName(styleName, lang)
    for familyNameElement in instanceElement.findall('familyname'):
        for key, lang in familyNameElement.items():
            if key == XML_LANG:
                familyName = familyNameElement.text
                instanceObject.setFamilyName(familyName, lang)
    for styleMapStyleNameElement in instanceElement.findall('stylemapstylename'):
        for key, lang in styleMapStyleNameElement.items():
            if key == XML_LANG:
                styleMapStyleName = styleMapStyleNameElement.text
                instanceObject.setStyleMapStyleName(styleMapStyleName, lang)
    for styleMapFamilyNameElement in instanceElement.findall('stylemapfamilyname'):
        for key, lang in styleMapFamilyNameElement.items():
            if key == XML_LANG:
                styleMapFamilyName = styleMapFamilyNameElement.text
                instanceObject.setStyleMapFamilyName(styleMapFamilyName, lang)
    designLocation, userLocation = self.locationFromElement(instanceElement)
    locationLabel = instanceElement.attrib.get('location')
    if (designLocation or userLocation) and locationLabel is not None:
        raise DesignSpaceDocumentError('instance element must have at most one of the location="..." attribute or the nested location element')
    instanceObject.locationLabel = locationLabel
    instanceObject.userLocation = userLocation or {}
    instanceObject.designLocation = designLocation or {}
    for glyphElement in instanceElement.findall('.glyphs/glyph'):
        self.readGlyphElement(glyphElement, instanceObject)
    for infoElement in instanceElement.findall('info'):
        self.readInfoElement(infoElement, instanceObject)
    for libElement in instanceElement.findall('lib'):
        self.readLibElement(libElement, instanceObject)
    self.documentObject.instances.append(instanceObject)