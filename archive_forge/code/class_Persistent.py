import os
import pickle
import sys
from zope.interface import Interface, implementer
from twisted.persisted import styles
from twisted.python import log, runtime
@implementer(IPersistable)
class Persistent:
    style = 'pickle'

    def __init__(self, original, name):
        self.original = original
        self.name = name

    def setStyle(self, style):
        """Set desired format.

        @type style: string (one of 'pickle' or 'source')
        """
        self.style = style

    def _getFilename(self, filename, ext, tag):
        if filename:
            finalname = filename
            filename = finalname + '-2'
        elif tag:
            filename = f'{self.name}-{tag}-2.{ext}'
            finalname = f'{self.name}-{tag}.{ext}'
        else:
            filename = f'{self.name}-2.{ext}'
            finalname = f'{self.name}.{ext}'
        return (finalname, filename)

    def _saveTemp(self, filename, dumpFunc):
        with open(filename, 'wb') as f:
            dumpFunc(self.original, f)

    def _getStyle(self):
        if self.style == 'source':
            from twisted.persisted.aot import jellyToSource as dumpFunc
            ext = 'tas'
        else:

            def dumpFunc(obj, file=None):
                pickle.dump(obj, file, 2)
            ext = 'tap'
        return (ext, dumpFunc)

    def save(self, tag=None, filename=None, passphrase=None):
        """Save object to file.

        @type tag: string
        @type filename: string
        @type passphrase: string
        """
        ext, dumpFunc = self._getStyle()
        if passphrase is not None:
            raise TypeError('passphrase must be None')
        finalname, filename = self._getFilename(filename, ext, tag)
        log.msg('Saving ' + self.name + ' application to ' + finalname + '...')
        self._saveTemp(filename, dumpFunc)
        if runtime.platformType == 'win32' and os.path.isfile(finalname):
            os.remove(finalname)
        os.rename(filename, finalname)
        log.msg('Saved.')