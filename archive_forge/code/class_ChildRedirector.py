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
class ChildRedirector(Redirect):
    isLeaf = False

    def __init__(self, url):
        if url.find('://') == -1 and (not url.startswith('..')) and (not url.startswith('/')):
            raise ValueError("It seems you've given me a redirect (%s) that is a child of myself! That's not good, it'll cause an infinite redirect." % url)
        Redirect.__init__(self, url)

    def getChild(self, name, request):
        newUrl = self.url
        if not newUrl.endswith('/'):
            newUrl += '/'
        newUrl += name
        return ChildRedirector(newUrl)