from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class BeforeHtmlPhase(Phase):
    __slots__ = tuple()

    def insertHtmlElement(self):
        self.tree.insertRoot(impliedTagToken('html', 'StartTag'))
        self.parser.phase = self.parser.phases['beforeHead']

    def processEOF(self):
        self.insertHtmlElement()
        return True

    def processComment(self, token):
        self.tree.insertComment(token, self.tree.document)

    def processSpaceCharacters(self, token):
        pass

    def processCharacters(self, token):
        self.insertHtmlElement()
        return token

    def processStartTag(self, token):
        if token['name'] == 'html':
            self.parser.firstStartTag = True
        self.insertHtmlElement()
        return token

    def processEndTag(self, token):
        if token['name'] not in ('head', 'body', 'html', 'br'):
            self.parser.parseError('unexpected-end-tag-before-html', {'name': token['name']})
        else:
            self.insertHtmlElement()
            return token