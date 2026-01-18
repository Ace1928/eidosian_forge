import os
from copy import deepcopy
from os import fsdecode
import logging
import zipfile
import enum
from collections import OrderedDict
import fs
import fs.base
import fs.subfs
import fs.errors
import fs.copy
import fs.osfs
import fs.zipfs
import fs.tempfs
import fs.tools
from fontTools.misc import plistlib
from fontTools.ufoLib.validators import *
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.converters import convertUFO1OrUFO2KerningToUFO3Kerning
from fontTools.ufoLib.errors import UFOLibError
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
def _convertFontInfoDataVersion3ToVersion2(data):
    converted = {}
    for attr, value in list(data.items()):
        newAttr, newValue = convertFontInfoValueForAttributeFromVersion3ToVersion2(attr, value)
        if newAttr not in fontInfoAttributesVersion2:
            continue
        converted[newAttr] = newValue
    return converted