from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class BeforeHeadPhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        self.startTagHead(impliedTagToken('head', 'StartTag'))
        return True

    def processSpaceCharacters(self, token):
        pass

    def processCharacters(self, token):
        self.startTagHead(impliedTagToken('head', 'StartTag'))
        return token

    def startTagHtml(self, token):
        return self.parser.phases['inBody'].processStartTag(token)

    def startTagHead(self, token):
        self.tree.insertElement(token)
        self.tree.headPointer = self.tree.openElements[-1]
        self.parser.phase = self.parser.phases['inHead']

    def startTagOther(self, token):
        self.startTagHead(impliedTagToken('head', 'StartTag'))
        return token

    def endTagImplyHead(self, token):
        self.startTagHead(impliedTagToken('head', 'StartTag'))
        return token

    def endTagOther(self, token):
        self.parser.parseError('end-tag-after-implied-root', {'name': token['name']})
    startTagHandler = _utils.MethodDispatcher([('html', startTagHtml), ('head', startTagHead)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([(('head', 'body', 'html', 'br'), endTagImplyHead)])
    endTagHandler.default = endTagOther