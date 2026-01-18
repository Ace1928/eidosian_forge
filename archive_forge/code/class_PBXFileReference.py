import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXFileReference(XCFileLikeElement, XCContainerPortal, XCRemoteObject):
    _schema = XCFileLikeElement._schema.copy()
    _schema.update({'explicitFileType': [0, str, 0, 0], 'lastKnownFileType': [0, str, 0, 0], 'name': [0, str, 0, 0], 'path': [0, str, 0, 1]})
    _should_print_single_line = True
    _encode_transforms = XCFileLikeElement._alternate_encode_transforms

    def __init__(self, properties=None, id=None, parent=None):
        XCFileLikeElement.__init__(self, properties, id, parent)
        if 'path' in self._properties and self._properties['path'].endswith('/'):
            self._properties['path'] = self._properties['path'][:-1]
            is_dir = True
        else:
            is_dir = False
        if 'path' in self._properties and 'lastKnownFileType' not in self._properties and ('explicitFileType' not in self._properties):
            extension_map = {'a': 'archive.ar', 'app': 'wrapper.application', 'bdic': 'file', 'bundle': 'wrapper.cfbundle', 'c': 'sourcecode.c.c', 'cc': 'sourcecode.cpp.cpp', 'cpp': 'sourcecode.cpp.cpp', 'css': 'text.css', 'cxx': 'sourcecode.cpp.cpp', 'dart': 'sourcecode', 'dylib': 'compiled.mach-o.dylib', 'framework': 'wrapper.framework', 'gyp': 'sourcecode', 'gypi': 'sourcecode', 'h': 'sourcecode.c.h', 'hxx': 'sourcecode.cpp.h', 'icns': 'image.icns', 'java': 'sourcecode.java', 'js': 'sourcecode.javascript', 'kext': 'wrapper.kext', 'm': 'sourcecode.c.objc', 'mm': 'sourcecode.cpp.objcpp', 'nib': 'wrapper.nib', 'o': 'compiled.mach-o.objfile', 'pdf': 'image.pdf', 'pl': 'text.script.perl', 'plist': 'text.plist.xml', 'pm': 'text.script.perl', 'png': 'image.png', 'py': 'text.script.python', 'r': 'sourcecode.rez', 'rez': 'sourcecode.rez', 's': 'sourcecode.asm', 'storyboard': 'file.storyboard', 'strings': 'text.plist.strings', 'swift': 'sourcecode.swift', 'ttf': 'file', 'xcassets': 'folder.assetcatalog', 'xcconfig': 'text.xcconfig', 'xcdatamodel': 'wrapper.xcdatamodel', 'xcdatamodeld': 'wrapper.xcdatamodeld', 'xib': 'file.xib', 'y': 'sourcecode.yacc'}
            prop_map = {'dart': 'explicitFileType', 'gyp': 'explicitFileType', 'gypi': 'explicitFileType'}
            if is_dir:
                file_type = 'folder'
                prop_name = 'lastKnownFileType'
            else:
                basename = posixpath.basename(self._properties['path'])
                root, ext = posixpath.splitext(basename)
                if ext != '':
                    ext = ext[1:].lower()
                file_type = extension_map.get(ext, 'text')
                prop_name = prop_map.get(ext, 'lastKnownFileType')
            self._properties[prop_name] = file_type