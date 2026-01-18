from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class AfterHeadPhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        self.anythingElse()
        return True

    def processCharacters(self, token):
        self.anythingElse()
        return token

    def startTagHtml(self, token):
        return self.parser.phases['inBody'].processStartTag(token)

    def startTagBody(self, token):
        self.parser.framesetOK = False
        self.tree.insertElement(token)
        self.parser.phase = self.parser.phases['inBody']

    def startTagFrameset(self, token):
        self.tree.insertElement(token)
        self.parser.phase = self.parser.phases['inFrameset']

    def startTagFromHead(self, token):
        self.parser.parseError('unexpected-start-tag-out-of-my-head', {'name': token['name']})
        self.tree.openElements.append(self.tree.headPointer)
        self.parser.phases['inHead'].processStartTag(token)
        for node in self.tree.openElements[::-1]:
            if node.name == 'head':
                self.tree.openElements.remove(node)
                break

    def startTagHead(self, token):
        self.parser.parseError('unexpected-start-tag', {'name': token['name']})

    def startTagOther(self, token):
        self.anythingElse()
        return token

    def endTagHtmlBodyBr(self, token):
        self.anythingElse()
        return token

    def endTagOther(self, token):
        self.parser.parseError('unexpected-end-tag', {'name': token['name']})

    def anythingElse(self):
        self.tree.insertElement(impliedTagToken('body', 'StartTag'))
        self.parser.phase = self.parser.phases['inBody']
        self.parser.framesetOK = True
    startTagHandler = _utils.MethodDispatcher([('html', startTagHtml), ('body', startTagBody), ('frameset', startTagFrameset), (('base', 'basefont', 'bgsound', 'link', 'meta', 'noframes', 'script', 'style', 'title'), startTagFromHead), ('head', startTagHead)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([(('body', 'html', 'br'), endTagHtmlBodyBr)])
    endTagHandler.default = endTagOther