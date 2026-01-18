from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InColumnGroupPhase(Phase):
    __slots__ = tuple()

    def ignoreEndTagColgroup(self):
        return self.tree.openElements[-1].name == 'html'

    def processEOF(self):
        if self.tree.openElements[-1].name == 'html':
            assert self.parser.innerHTML
            return
        else:
            ignoreEndTag = self.ignoreEndTagColgroup()
            self.endTagColgroup(impliedTagToken('colgroup'))
            if not ignoreEndTag:
                return True

    def processCharacters(self, token):
        ignoreEndTag = self.ignoreEndTagColgroup()
        self.endTagColgroup(impliedTagToken('colgroup'))
        if not ignoreEndTag:
            return token

    def startTagCol(self, token):
        self.tree.insertElement(token)
        self.tree.openElements.pop()
        token['selfClosingAcknowledged'] = True

    def startTagOther(self, token):
        ignoreEndTag = self.ignoreEndTagColgroup()
        self.endTagColgroup(impliedTagToken('colgroup'))
        if not ignoreEndTag:
            return token

    def endTagColgroup(self, token):
        if self.ignoreEndTagColgroup():
            assert self.parser.innerHTML
            self.parser.parseError()
        else:
            self.tree.openElements.pop()
            self.parser.phase = self.parser.phases['inTable']

    def endTagCol(self, token):
        self.parser.parseError('no-end-tag', {'name': 'col'})

    def endTagOther(self, token):
        ignoreEndTag = self.ignoreEndTagColgroup()
        self.endTagColgroup(impliedTagToken('colgroup'))
        if not ignoreEndTag:
            return token
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), ('col', startTagCol)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('colgroup', endTagColgroup), ('col', endTagCol)])
    endTagHandler.default = endTagOther