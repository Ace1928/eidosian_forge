import io
import linecache
import warnings
from collections import OrderedDict
from html import escape
from typing import (
from xml.sax import handler, make_parser
from xml.sax.xmlreader import AttributesNSImpl, Locator
from zope.interface import implementer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import urlpath
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import fullyQualifiedName
from twisted.web import resource
from twisted.web._element import Element, renderer
from twisted.web._flatten import Flattenable, flatten, flattenString
from twisted.web._stan import CDATA, Comment, Tag, slot
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
class _SourceFragmentElement(Element):
    """
    L{_SourceFragmentElement} is an L{IRenderable} which can render several lines
    of source code near the line number of a particular frame object.

    @ivar frame: A L{Failure<twisted.python.failure.Failure>}-style frame object
        for which to load a source line to render.  This is really a tuple
        holding some information from a frame object.  See
        L{Failure.frames<twisted.python.failure.Failure>} for specifics.
    """

    def __init__(self, loader, frame):
        Element.__init__(self, loader)
        self.frame = frame

    def _getSourceLines(self):
        """
        Find the source line references by C{self.frame} and yield, in source
        line order, it and the previous and following lines.

        @return: A generator which yields two-tuples.  Each tuple gives a source
            line number and the contents of that source line.
        """
        filename = self.frame[1]
        lineNumber = self.frame[2]
        for snipLineNumber in range(lineNumber - 1, lineNumber + 2):
            yield (snipLineNumber, linecache.getline(filename, snipLineNumber).rstrip())

    @renderer
    def sourceLines(self, request, tag):
        """
        Render the source line indicated by C{self.frame} and several
        surrounding lines.  The active line will be given a I{class} of
        C{"snippetHighlightLine"}.  Other lines will be given a I{class} of
        C{"snippetLine"}.
        """
        for lineNumber, sourceLine in self._getSourceLines():
            newTag = tag.clone()
            if lineNumber == self.frame[2]:
                cssClass = 'snippetHighlightLine'
            else:
                cssClass = 'snippetLine'
            loader = TagLoader(newTag(**{'class': cssClass}))
            yield _SourceLineElement(loader, lineNumber, sourceLine)