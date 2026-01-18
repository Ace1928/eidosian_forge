from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InHeadNoscriptPhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        self.parser.parseError('eof-in-head-noscript')
        self.anythingElse()
        return True

    def processComment(self, token):
        return self.parser.phases['inHead'].processComment(token)

    def processCharacters(self, token):
        self.parser.parseError('char-in-head-noscript')
        self.anythingElse()
        return token

    def processSpaceCharacters(self, token):
        return self.parser.phases['inHead'].processSpaceCharacters(token)

    def startTagHtml(self, token):
        return self.parser.phases['inBody'].processStartTag(token)

    def startTagBaseLinkCommand(self, token):
        return self.parser.phases['inHead'].processStartTag(token)

    def startTagHeadNoscript(self, token):
        self.parser.parseError('unexpected-start-tag', {'name': token['name']})

    def startTagOther(self, token):
        self.parser.parseError('unexpected-inhead-noscript-tag', {'name': token['name']})
        self.anythingElse()
        return token

    def endTagNoscript(self, token):
        node = self.parser.tree.openElements.pop()
        assert node.name == 'noscript', 'Expected noscript got %s' % node.name
        self.parser.phase = self.parser.phases['inHead']

    def endTagBr(self, token):
        self.parser.parseError('unexpected-inhead-noscript-tag', {'name': token['name']})
        self.anythingElse()
        return token

    def endTagOther(self, token):
        self.parser.parseError('unexpected-end-tag', {'name': token['name']})

    def anythingElse(self):
        self.endTagNoscript(impliedTagToken('noscript'))
    startTagHandler = _utils.MethodDispatcher([('html', startTagHtml), (('basefont', 'bgsound', 'link', 'meta', 'noframes', 'style'), startTagBaseLinkCommand), (('head', 'noscript'), startTagHeadNoscript)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('noscript', endTagNoscript), ('br', endTagBr)])
    endTagHandler.default = endTagOther