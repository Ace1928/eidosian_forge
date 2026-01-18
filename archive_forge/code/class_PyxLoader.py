imports, including parts of the standard library and installed
import glob
import imp
import os
import sys
from zipimport import zipimporter, ZipImportError
class PyxLoader(object):

    def __init__(self, fullname, path, init_path=None, pyxbuild_dir=None, inplace=False, language_level=None):
        _debug('PyxLoader created for loading %s from %s (init path: %s)', fullname, path, init_path)
        self.fullname = fullname
        self.path, self.init_path = (path, init_path)
        self.pyxbuild_dir = pyxbuild_dir
        self.inplace = inplace
        self.language_level = language_level

    def load_module(self, fullname):
        assert self.fullname == fullname, 'invalid module, expected %s, got %s' % (self.fullname, fullname)
        if self.init_path:
            module = load_module(fullname, self.init_path, self.pyxbuild_dir, is_package=True, build_inplace=self.inplace, language_level=self.language_level)
            module.__path__ = [self.path]
        else:
            module = load_module(fullname, self.path, self.pyxbuild_dir, build_inplace=self.inplace, language_level=self.language_level)
        return module