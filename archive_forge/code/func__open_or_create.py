from __future__ import absolute_import, unicode_literals
import io
import posixpath
import sys
from os import environ
from pybtex.exceptions import PybtexError
from pybtex.kpathsea import kpsewhich
def _open_or_create(opener, filename, mode, environ, **kwargs):
    try:
        return opener(filename, mode, **kwargs)
    except EnvironmentError as error:
        if 'TEXMFOUTPUT' in environ:
            new_filename = posixpath.join(environ['TEXMFOUTPUT'], filename)
            try:
                return opener(new_filename, mode, **kwargs)
            except EnvironmentError:
                pass
        raise error