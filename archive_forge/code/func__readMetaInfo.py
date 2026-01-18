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
def _readMetaInfo(self, validate=None):
    """
        Read metainfo.plist and return raw data. Only used for internal operations.

        ``validate`` will validate the read data, by default it is set
        to the class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    data = self._getPlist(METAINFO_FILENAME)
    if validate and (not isinstance(data, dict)):
        raise UFOLibError('metainfo.plist is not properly formatted.')
    try:
        formatVersionMajor = data['formatVersion']
    except KeyError:
        raise UFOLibError(f"Missing required formatVersion in '{METAINFO_FILENAME}' on {self.fs}")
    formatVersionMinor = data.setdefault('formatVersionMinor', 0)
    try:
        formatVersion = UFOFormatVersion((formatVersionMajor, formatVersionMinor))
    except ValueError as e:
        unsupportedMsg = f"Unsupported UFO format ({formatVersionMajor}.{formatVersionMinor}) in '{METAINFO_FILENAME}' on {self.fs}"
        if validate:
            from fontTools.ufoLib.errors import UnsupportedUFOFormat
            raise UnsupportedUFOFormat(unsupportedMsg) from e
        formatVersion = UFOFormatVersion.default()
        logger.warning('%s. Assuming the latest supported version (%s). Some data may be skipped or parsed incorrectly', unsupportedMsg, formatVersion)
    data['formatVersionTuple'] = formatVersion
    return data