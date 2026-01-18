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
class _FrameElement(Element):
    """
    L{_FrameElement} is an L{IRenderable} which can render details about one
    frame from a L{Failure<twisted.python.failure.Failure>}.

    @ivar frame: A L{Failure<twisted.python.failure.Failure>}-style frame object
        for which to load a source line to render.  This is really a tuple
        holding some information from a frame object.  See
        L{Failure.frames<twisted.python.failure.Failure>} for specifics.
    """

    def __init__(self, loader, frame):
        Element.__init__(self, loader)
        self.frame = frame

    @renderer
    def filename(self, request, tag):
        """
        Render the name of the file this frame references as a child of C{tag}.
        """
        return tag(self.frame[1])

    @renderer
    def lineNumber(self, request, tag):
        """
        Render the source line number this frame references as a child of
        C{tag}.
        """
        return tag(str(self.frame[2]))

    @renderer
    def function(self, request, tag):
        """
        Render the function name this frame references as a child of C{tag}.
        """
        return tag(self.frame[0])

    @renderer
    def source(self, request, tag):
        """
        Render the source code surrounding the line this frame references,
        replacing C{tag}.
        """
        return _SourceFragmentElement(TagLoader(tag), self.frame)