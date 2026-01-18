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
class UFOReader(_UFOBaseIO):
    """
    Read the various components of the .ufo.

    By default read data is validated. Set ``validate`` to
    ``False`` to not validate the data.
    """

    def __init__(self, path, validate=True):
        if hasattr(path, '__fspath__'):
            path = path.__fspath__()
        if isinstance(path, str):
            structure = _sniffFileStructure(path)
            try:
                if structure is UFOFileStructure.ZIP:
                    parentFS = fs.zipfs.ZipFS(path, write=False, encoding='utf-8')
                else:
                    parentFS = fs.osfs.OSFS(path)
            except fs.errors.CreateFailed as e:
                raise UFOLibError(f"unable to open '{path}': {e}")
            if structure is UFOFileStructure.ZIP:
                rootDirs = [p.name for p in parentFS.scandir('/') if p.is_dir and p.name != '__MACOSX']
                if len(rootDirs) == 1:
                    self.fs = parentFS.opendir(rootDirs[0], factory=fs.subfs.ClosingSubFS)
                else:
                    raise UFOLibError('Expected exactly 1 root directory, found %d' % len(rootDirs))
            else:
                self.fs = parentFS
            self._shouldClose = True
            self._fileStructure = structure
        elif isinstance(path, fs.base.FS):
            filesystem = path
            try:
                filesystem.check()
            except fs.errors.FilesystemClosed:
                raise UFOLibError("the filesystem '%s' is closed" % path)
            else:
                self.fs = filesystem
            try:
                path = filesystem.getsyspath('/')
            except fs.errors.NoSysPath:
                path = str(filesystem)
            self._shouldClose = False
            self._fileStructure = UFOFileStructure.PACKAGE
        else:
            raise TypeError("Expected a path string or fs.base.FS object, found '%s'" % type(path).__name__)
        self._path = fsdecode(path)
        self._validate = validate
        self._upConvertedKerningData = None
        try:
            self.readMetaInfo(validate=validate)
        except UFOLibError:
            self.close()
            raise

    def _get_path(self):
        import warnings
        warnings.warn("The 'path' attribute is deprecated; use the 'fs' attribute instead", DeprecationWarning, stacklevel=2)
        return self._path
    path = property(_get_path, doc='The path of the UFO (DEPRECATED).')

    def _get_formatVersion(self):
        import warnings
        warnings.warn("The 'formatVersion' attribute is deprecated; use the 'formatVersionTuple'", DeprecationWarning, stacklevel=2)
        return self._formatVersion.major
    formatVersion = property(_get_formatVersion, doc='The (major) format version of the UFO. DEPRECATED: Use formatVersionTuple')

    @property
    def formatVersionTuple(self):
        """The (major, minor) format version of the UFO.
        This is determined by reading metainfo.plist during __init__.
        """
        return self._formatVersion

    def _get_fileStructure(self):
        return self._fileStructure
    fileStructure = property(_get_fileStructure, doc='The file structure of the UFO: either UFOFileStructure.ZIP or UFOFileStructure.PACKAGE')

    def _upConvertKerning(self, validate):
        """
        Up convert kerning and groups in UFO 1 and 2.
        The data will be held internally until each bit of data
        has been retrieved. The conversion of both must be done
        at once, so the raw data is cached and an error is raised
        if one bit of data becomes obsolete before it is called.

        ``validate`` will validate the data.
        """
        if self._upConvertedKerningData:
            testKerning = self._readKerning()
            if testKerning != self._upConvertedKerningData['originalKerning']:
                raise UFOLibError('The data in kerning.plist has been modified since it was converted to UFO 3 format.')
            testGroups = self._readGroups()
            if testGroups != self._upConvertedKerningData['originalGroups']:
                raise UFOLibError('The data in groups.plist has been modified since it was converted to UFO 3 format.')
        else:
            groups = self._readGroups()
            if validate:
                invalidFormatMessage = 'groups.plist is not properly formatted.'
                if not isinstance(groups, dict):
                    raise UFOLibError(invalidFormatMessage)
                for groupName, glyphList in groups.items():
                    if not isinstance(groupName, str):
                        raise UFOLibError(invalidFormatMessage)
                    elif not isinstance(glyphList, list):
                        raise UFOLibError(invalidFormatMessage)
                    for glyphName in glyphList:
                        if not isinstance(glyphName, str):
                            raise UFOLibError(invalidFormatMessage)
            self._upConvertedKerningData = dict(kerning={}, originalKerning=self._readKerning(), groups={}, originalGroups=groups)
            kerning, groups, conversionMaps = convertUFO1OrUFO2KerningToUFO3Kerning(self._upConvertedKerningData['originalKerning'], deepcopy(self._upConvertedKerningData['originalGroups']), self.getGlyphSet())
            self._upConvertedKerningData['kerning'] = kerning
            self._upConvertedKerningData['groups'] = groups
            self._upConvertedKerningData['groupRenameMaps'] = conversionMaps

    def readBytesFromPath(self, path):
        """
        Returns the bytes in the file at the given path.
        The path must be relative to the UFO's filesystem root.
        Returns None if the file does not exist.
        """
        try:
            return self.fs.readbytes(fsdecode(path))
        except fs.errors.ResourceNotFound:
            return None

    def getReadFileForPath(self, path, encoding=None):
        """
        Returns a file (or file-like) object for the file at the given path.
        The path must be relative to the UFO path.
        Returns None if the file does not exist.
        By default the file is opened in binary mode (reads bytes).
        If encoding is passed, the file is opened in text mode (reads str).

        Note: The caller is responsible for closing the open file.
        """
        path = fsdecode(path)
        try:
            if encoding is None:
                return self.fs.openbin(path)
            else:
                return self.fs.open(path, mode='r', encoding=encoding)
        except fs.errors.ResourceNotFound:
            return None

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

    def readMetaInfo(self, validate=None):
        """
        Read metainfo.plist and set formatVersion. Only used for internal operations.

        ``validate`` will validate the read data, by default it is set
        to the class's validate value, can be overridden.
        """
        data = self._readMetaInfo(validate=validate)
        self._formatVersion = data['formatVersionTuple']

    def _readGroups(self):
        groups = self._getPlist(GROUPS_FILENAME, {})
        for groupName, glyphList in groups.items():
            if groupName.startswith(('public.kern1.', 'public.kern2.')):
                groups[groupName] = list(OrderedDict.fromkeys(glyphList))
        return groups

    def readGroups(self, validate=None):
        """
        Read groups.plist. Returns a dict.
        ``validate`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            self._upConvertKerning(validate)
            groups = self._upConvertedKerningData['groups']
        else:
            groups = self._readGroups()
        if validate:
            valid, message = groupsValidator(groups)
            if not valid:
                raise UFOLibError(message)
        return groups

    def getKerningGroupConversionRenameMaps(self, validate=None):
        """
        Get maps defining the renaming that was done during any
        needed kerning group conversion. This method returns a
        dictionary of this form::

                {
                        "side1" : {"old group name" : "new group name"},
                        "side2" : {"old group name" : "new group name"}
                }

        When no conversion has been performed, the side1 and side2
        dictionaries will be empty.

        ``validate`` will validate the groups, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion >= UFOFormatVersion.FORMAT_3_0:
            return dict(side1={}, side2={})
        self.readGroups(validate=validate)
        return self._upConvertedKerningData['groupRenameMaps']

    def _readInfo(self, validate):
        data = self._getPlist(FONTINFO_FILENAME, {})
        if validate and (not isinstance(data, dict)):
            raise UFOLibError('fontinfo.plist is not properly formatted.')
        return data

    def readInfo(self, info, validate=None):
        """
        Read fontinfo.plist. It requires an object that allows
        setting attributes with names that follow the fontinfo.plist
        version 3 specification. This will write the attributes
        defined in the file into the object.

        ``validate`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        infoDict = self._readInfo(validate)
        infoDataToSet = {}
        if self._formatVersion == UFOFormatVersion.FORMAT_1_0:
            for attr in fontInfoAttributesVersion1:
                value = infoDict.get(attr)
                if value is not None:
                    infoDataToSet[attr] = value
            infoDataToSet = _convertFontInfoDataVersion1ToVersion2(infoDataToSet)
            infoDataToSet = _convertFontInfoDataVersion2ToVersion3(infoDataToSet)
        elif self._formatVersion == UFOFormatVersion.FORMAT_2_0:
            for attr, dataValidationDict in list(fontInfoAttributesVersion2ValueData.items()):
                value = infoDict.get(attr)
                if value is None:
                    continue
                infoDataToSet[attr] = value
            infoDataToSet = _convertFontInfoDataVersion2ToVersion3(infoDataToSet)
        elif self._formatVersion.major == UFOFormatVersion.FORMAT_3_0.major:
            for attr, dataValidationDict in list(fontInfoAttributesVersion3ValueData.items()):
                value = infoDict.get(attr)
                if value is None:
                    continue
                infoDataToSet[attr] = value
        else:
            raise NotImplementedError(self._formatVersion)
        if validate:
            infoDataToSet = validateInfoVersion3Data(infoDataToSet)
        for attr, value in list(infoDataToSet.items()):
            try:
                setattr(info, attr, value)
            except AttributeError:
                raise UFOLibError('The supplied info object does not support setting a necessary attribute (%s).' % attr)

    def _readKerning(self):
        data = self._getPlist(KERNING_FILENAME, {})
        return data

    def readKerning(self, validate=None):
        """
        Read kerning.plist. Returns a dict.

        ``validate`` will validate the kerning data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            self._upConvertKerning(validate)
            kerningNested = self._upConvertedKerningData['kerning']
        else:
            kerningNested = self._readKerning()
        if validate:
            valid, message = kerningValidator(kerningNested)
            if not valid:
                raise UFOLibError(message)
        kerning = {}
        for left in kerningNested:
            for right in kerningNested[left]:
                value = kerningNested[left][right]
                kerning[left, right] = value
        return kerning

    def readLib(self, validate=None):
        """
        Read lib.plist. Returns a dict.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        data = self._getPlist(LIB_FILENAME, {})
        if validate:
            valid, message = fontLibValidator(data)
            if not valid:
                raise UFOLibError(message)
        return data

    def readFeatures(self):
        """
        Read features.fea. Return a string.
        The returned string is empty if the file is missing.
        """
        try:
            with self.fs.open(FEATURES_FILENAME, 'r', encoding='utf-8') as f:
                return f.read()
        except fs.errors.ResourceNotFound:
            return ''

    def _readLayerContents(self, validate):
        """
        Rebuild the layer contents list by checking what glyphsets
        are available on disk.

        ``validate`` will validate the layer contents.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return [(DEFAULT_LAYER_NAME, DEFAULT_GLYPHS_DIRNAME)]
        contents = self._getPlist(LAYERCONTENTS_FILENAME)
        if validate:
            valid, error = layerContentsValidator(contents, self.fs)
            if not valid:
                raise UFOLibError(error)
        return contents

    def getLayerNames(self, validate=None):
        """
        Get the ordered layer names from layercontents.plist.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        layerContents = self._readLayerContents(validate)
        layerNames = [layerName for layerName, directoryName in layerContents]
        return layerNames

    def getDefaultLayerName(self, validate=None):
        """
        Get the default layer name from layercontents.plist.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        layerContents = self._readLayerContents(validate)
        for layerName, layerDirectory in layerContents:
            if layerDirectory == DEFAULT_GLYPHS_DIRNAME:
                return layerName
        raise UFOLibError('The default layer is not defined in layercontents.plist.')

    def getGlyphSet(self, layerName=None, validateRead=None, validateWrite=None):
        """
        Return the GlyphSet associated with the
        glyphs directory mapped to layerName
        in the UFO. If layerName is not provided,
        the name retrieved with getDefaultLayerName
        will be used.

        ``validateRead`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        ``validateWrite`` will validate the written data, by default it is set to the
        class's validate value, can be overridden.
        """
        from fontTools.ufoLib.glifLib import GlyphSet
        if validateRead is None:
            validateRead = self._validate
        if validateWrite is None:
            validateWrite = self._validate
        if layerName is None:
            layerName = self.getDefaultLayerName(validate=validateRead)
        directory = None
        layerContents = self._readLayerContents(validateRead)
        for storedLayerName, storedLayerDirectory in layerContents:
            if layerName == storedLayerName:
                directory = storedLayerDirectory
                break
        if directory is None:
            raise UFOLibError('No glyphs directory is mapped to "%s".' % layerName)
        try:
            glyphSubFS = self.fs.opendir(directory)
        except fs.errors.ResourceNotFound:
            raise UFOLibError(f"No '{directory}' directory for layer '{layerName}'")
        return GlyphSet(glyphSubFS, ufoFormatVersion=self._formatVersion, validateRead=validateRead, validateWrite=validateWrite, expectContentsFile=True)

    def getCharacterMapping(self, layerName=None, validate=None):
        """
        Return a dictionary that maps unicode values (ints) to
        lists of glyph names.
        """
        if validate is None:
            validate = self._validate
        glyphSet = self.getGlyphSet(layerName, validateRead=validate, validateWrite=True)
        allUnicodes = glyphSet.getUnicodes()
        cmap = {}
        for glyphName, unicodes in allUnicodes.items():
            for code in unicodes:
                if code in cmap:
                    cmap[code].append(glyphName)
                else:
                    cmap[code] = [glyphName]
        return cmap

    def getDataDirectoryListing(self):
        """
        Returns a list of all files in the data directory.
        The returned paths will be relative to the UFO.
        This will not list directory names, only file names.
        Thus, empty directories will be skipped.
        """
        try:
            self._dataFS = self.fs.opendir(DATA_DIRNAME)
        except fs.errors.ResourceNotFound:
            return []
        except fs.errors.DirectoryExpected:
            raise UFOLibError('The UFO contains a "data" file instead of a directory.')
        try:
            return [p.lstrip('/') for p in self._dataFS.walk.files()]
        except fs.errors.ResourceError:
            return []

    def getImageDirectoryListing(self, validate=None):
        """
        Returns a list of all image file names in
        the images directory. Each of the images will
        have been verified to have the PNG signature.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return []
        if validate is None:
            validate = self._validate
        try:
            self._imagesFS = imagesFS = self.fs.opendir(IMAGES_DIRNAME)
        except fs.errors.ResourceNotFound:
            return []
        except fs.errors.DirectoryExpected:
            raise UFOLibError('The UFO contains an "images" file instead of a directory.')
        result = []
        for path in imagesFS.scandir('/'):
            if path.is_dir:
                continue
            if validate:
                with imagesFS.openbin(path.name) as fp:
                    valid, error = pngValidator(fileObj=fp)
                if valid:
                    result.append(path.name)
            else:
                result.append(path.name)
        return result

    def readData(self, fileName):
        """
        Return bytes for the file named 'fileName' inside the 'data/' directory.
        """
        fileName = fsdecode(fileName)
        try:
            try:
                dataFS = self._dataFS
            except AttributeError:
                dataFS = self.fs.opendir(DATA_DIRNAME)
            data = dataFS.readbytes(fileName)
        except fs.errors.ResourceNotFound:
            raise UFOLibError(f"No data file named '{fileName}' on {self.fs}")
        return data

    def readImage(self, fileName, validate=None):
        """
        Return image data for the file named fileName.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            raise UFOLibError(f'Reading images is not allowed in UFO {self._formatVersion.major}.')
        fileName = fsdecode(fileName)
        try:
            try:
                imagesFS = self._imagesFS
            except AttributeError:
                imagesFS = self.fs.opendir(IMAGES_DIRNAME)
            data = imagesFS.readbytes(fileName)
        except fs.errors.ResourceNotFound:
            raise UFOLibError(f"No image file named '{fileName}' on {self.fs}")
        if validate:
            valid, error = pngValidator(data=data)
            if not valid:
                raise UFOLibError(error)
        return data

    def close(self):
        if self._shouldClose:
            self.fs.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()