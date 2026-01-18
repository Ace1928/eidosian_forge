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
def _getGlyphSetFormatVersion3(self, validateRead, validateWrite, layerName=None, defaultLayer=True, glyphNameToFileNameFunc=None, expectContentsFile=False):
    from fontTools.ufoLib.glifLib import GlyphSet
    if defaultLayer:
        for existingLayerName, directory in self.layerContents.items():
            if directory == DEFAULT_GLYPHS_DIRNAME:
                if existingLayerName != layerName:
                    raise UFOLibError("Another layer ('%s') is already mapped to the default directory." % existingLayerName)
            elif existingLayerName == layerName:
                raise UFOLibError('The layer name is already mapped to a non-default layer.')
    if layerName in self.layerContents:
        directory = self.layerContents[layerName]
    elif defaultLayer:
        directory = DEFAULT_GLYPHS_DIRNAME
    else:
        existing = {d.lower() for d in self.layerContents.values()}
        directory = userNameToFileName(layerName, existing=existing, prefix='glyphs.')
    glyphSubFS = self.fs.makedir(directory, recreate=True)
    self.layerContents[layerName] = directory
    return GlyphSet(glyphSubFS, glyphNameToFileNameFunc=glyphNameToFileNameFunc, ufoFormatVersion=self._formatVersion, validateRead=validateRead, validateWrite=validateWrite, expectContentsFile=expectContentsFile)