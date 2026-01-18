from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class AfterBodyPhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        pass

    def processComment(self, token):
        self.tree.insertComment(token, self.tree.openElements[0])

    def processCharacters(self, token):
        self.parser.parseError('unexpected-char-after-body')
        self.parser.phase = self.parser.phases['inBody']
        return token

    def startTagHtml(self, token):
        return self.parser.phases['inBody'].processStartTag(token)

    def startTagOther(self, token):
        self.parser.parseError('unexpected-start-tag-after-body', {'name': token['name']})
        self.parser.phase = self.parser.phases['inBody']
        return token

    def endTagHtml(self, name):
        if self.parser.innerHTML:
            self.parser.parseError('unexpected-end-tag-after-body-innerhtml')
        else:
            self.parser.phase = self.parser.phases['afterAfterBody']

    def endTagOther(self, token):
        self.parser.parseError('unexpected-end-tag-after-body', {'name': token['name']})
        self.parser.phase = self.parser.phases['inBody']
        return token
    startTagHandler = _utils.MethodDispatcher([('html', startTagHtml)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('html', endTagHtml)])
    endTagHandler.default = endTagOther