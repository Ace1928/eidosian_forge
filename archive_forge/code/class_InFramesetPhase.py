from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InFramesetPhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        if self.tree.openElements[-1].name != 'html':
            self.parser.parseError('eof-in-frameset')
        else:
            assert self.parser.innerHTML

    def processCharacters(self, token):
        self.parser.parseError('unexpected-char-in-frameset')

    def startTagFrameset(self, token):
        self.tree.insertElement(token)

    def startTagFrame(self, token):
        self.tree.insertElement(token)
        self.tree.openElements.pop()

    def startTagNoframes(self, token):
        return self.parser.phases['inBody'].processStartTag(token)

    def startTagOther(self, token):
        self.parser.parseError('unexpected-start-tag-in-frameset', {'name': token['name']})

    def endTagFrameset(self, token):
        if self.tree.openElements[-1].name == 'html':
            self.parser.parseError('unexpected-frameset-in-frameset-innerhtml')
        else:
            self.tree.openElements.pop()
        if not self.parser.innerHTML and self.tree.openElements[-1].name != 'frameset':
            self.parser.phase = self.parser.phases['afterFrameset']

    def endTagOther(self, token):
        self.parser.parseError('unexpected-end-tag-in-frameset', {'name': token['name']})
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), ('frameset', startTagFrameset), ('frame', startTagFrame), ('noframes', startTagNoframes)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('frameset', endTagFrameset)])
    endTagHandler.default = endTagOther