import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase
class XMLWriter:
    """
    Parameters
    ----------
    file : writable text file-like object
    """

    def __init__(self, file):
        self.__write = file.write
        if hasattr(file, 'flush'):
            self.flush = file.flush
        self.__open = 0
        self.__tags = []
        self.__data = []
        self.__indentation = ' ' * 64

    def __flush(self, indent=True):
        if self.__open:
            if indent:
                self.__write('>\n')
            else:
                self.__write('>')
            self.__open = 0
        if self.__data:
            data = ''.join(self.__data)
            self.__write(_escape_cdata(data))
            self.__data = []

    def start(self, tag, attrib={}, **extra):
        """
        Open a new element.  Attributes can be given as keyword
        arguments, or as a string/string dictionary. The method returns
        an opaque identifier that can be passed to the :meth:`close`
        method, to close all open elements up to and including this one.

        Parameters
        ----------
        tag
            Element tag.
        attrib
            Attribute dictionary.  Alternatively, attributes can be given as
            keyword arguments.

        Returns
        -------
        An element identifier.
        """
        self.__flush()
        tag = _escape_cdata(tag)
        self.__data = []
        self.__tags.append(tag)
        self.__write(self.__indentation[:len(self.__tags) - 1])
        self.__write(f'<{tag}')
        for k, v in {**attrib, **extra}.items():
            if v:
                k = _escape_cdata(k)
                v = _quote_escape_attrib(v)
                self.__write(f' {k}={v}')
        self.__open = 1
        return len(self.__tags) - 1

    def comment(self, comment):
        """
        Add a comment to the output stream.

        Parameters
        ----------
        comment : str
            Comment text.
        """
        self.__flush()
        self.__write(self.__indentation[:len(self.__tags)])
        self.__write(f'<!-- {_escape_comment(comment)} -->\n')

    def data(self, text):
        """
        Add character data to the output stream.

        Parameters
        ----------
        text : str
            Character data.
        """
        self.__data.append(text)

    def end(self, tag=None, indent=True):
        """
        Close the current element (opened by the most recent call to
        :meth:`start`).

        Parameters
        ----------
        tag
            Element tag.  If given, the tag must match the start tag.  If
            omitted, the current element is closed.
        indent : bool, default: True
        """
        if tag:
            assert self.__tags, f'unbalanced end({tag})'
            assert _escape_cdata(tag) == self.__tags[-1], f'expected end({self.__tags[-1]}), got {tag}'
        else:
            assert self.__tags, 'unbalanced end()'
        tag = self.__tags.pop()
        if self.__data:
            self.__flush(indent)
        elif self.__open:
            self.__open = 0
            self.__write('/>\n')
            return
        if indent:
            self.__write(self.__indentation[:len(self.__tags)])
        self.__write(f'</{tag}>\n')

    def close(self, id):
        """
        Close open elements, up to (and including) the element identified
        by the given identifier.

        Parameters
        ----------
        id
            Element identifier, as returned by the :meth:`start` method.
        """
        while len(self.__tags) > id:
            self.end()

    def element(self, tag, text=None, attrib={}, **extra):
        """
        Add an entire element.  This is the same as calling :meth:`start`,
        :meth:`data`, and :meth:`end` in sequence. The *text* argument can be
        omitted.
        """
        self.start(tag, attrib, **extra)
        if text:
            self.data(text)
        self.end(indent=False)

    def flush(self):
        """Flush the output stream."""
        pass