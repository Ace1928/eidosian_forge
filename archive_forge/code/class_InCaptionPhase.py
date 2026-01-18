from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InCaptionPhase(Phase):
    __slots__ = tuple()

    def ignoreEndTagCaption(self):
        return not self.tree.elementInScope('caption', variant='table')

    def processEOF(self):
        self.parser.phases['inBody'].processEOF()

    def processCharacters(self, token):
        return self.parser.phases['inBody'].processCharacters(token)

    def startTagTableElement(self, token):
        self.parser.parseError()
        ignoreEndTag = self.ignoreEndTagCaption()
        self.parser.phase.processEndTag(impliedTagToken('caption'))
        if not ignoreEndTag:
            return token

    def startTagOther(self, token):
        return self.parser.phases['inBody'].processStartTag(token)

    def endTagCaption(self, token):
        if not self.ignoreEndTagCaption():
            self.tree.generateImpliedEndTags()
            if self.tree.openElements[-1].name != 'caption':
                self.parser.parseError('expected-one-end-tag-but-got-another', {'gotName': 'caption', 'expectedName': self.tree.openElements[-1].name})
            while self.tree.openElements[-1].name != 'caption':
                self.tree.openElements.pop()
            self.tree.openElements.pop()
            self.tree.clearActiveFormattingElements()
            self.parser.phase = self.parser.phases['inTable']
        else:
            assert self.parser.innerHTML
            self.parser.parseError()

    def endTagTable(self, token):
        self.parser.parseError()
        ignoreEndTag = self.ignoreEndTagCaption()
        self.parser.phase.processEndTag(impliedTagToken('caption'))
        if not ignoreEndTag:
            return token

    def endTagIgnore(self, token):
        self.parser.parseError('unexpected-end-tag', {'name': token['name']})

    def endTagOther(self, token):
        return self.parser.phases['inBody'].processEndTag(token)
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), (('caption', 'col', 'colgroup', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), startTagTableElement)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('caption', endTagCaption), ('table', endTagTable), (('body', 'col', 'colgroup', 'html', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), endTagIgnore)])
    endTagHandler.default = endTagOther