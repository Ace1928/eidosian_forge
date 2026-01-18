from OpenGL.latebind import LateBind
from OpenGL._bytes import bytes,unicode,as_8_bit
import OpenGL as root
import sys
import logging
class ExtensionQuerier(object):
    prefix = None
    version_prefix = None
    assumed_version = [1, 0]
    version = extensions = None
    version_string = extensions_string = None
    registered = []

    def __init__(self):
        self.registered.append(self)

    @classmethod
    def hasExtension(self, specifier):
        for registered in self.registered:
            result = registered(specifier)
            if result:
                return result
        return False

    def __call__(self, specifier):
        specifier = as_8_bit(specifier).replace(as_8_bit('.'), as_8_bit('_'))
        if not specifier.startswith(as_8_bit(self.prefix)):
            return None
        if specifier.startswith(as_8_bit(self.version_prefix)):
            specifier = [int(x) for x in specifier[len(self.version_prefix):].split(as_8_bit('_'))]
            if specifier[:2] <= self.assumed_version:
                return True
            version = self.getVersion()
            if not version:
                return version
            return specifier <= version
        else:
            extensions = self.getExtensions()
            return extensions and specifier in extensions

    def getVersion(self):
        if not self.version:
            self.version = self.pullVersion()
        return self.version

    def getExtensions(self):
        if not self.extensions:
            self.extensions = self.pullExtensions()
        return self.extensions