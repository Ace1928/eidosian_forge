from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InCellPhase(Phase):
    __slots__ = tuple()

    def closeCell(self):
        if self.tree.elementInScope('td', variant='table'):
            self.endTagTableCell(impliedTagToken('td'))
        elif self.tree.elementInScope('th', variant='table'):
            self.endTagTableCell(impliedTagToken('th'))

    def processEOF(self):
        self.parser.phases['inBody'].processEOF()

    def processCharacters(self, token):
        return self.parser.phases['inBody'].processCharacters(token)

    def startTagTableOther(self, token):
        if self.tree.elementInScope('td', variant='table') or self.tree.elementInScope('th', variant='table'):
            self.closeCell()
            return token
        else:
            assert self.parser.innerHTML
            self.parser.parseError()

    def startTagOther(self, token):
        return self.parser.phases['inBody'].processStartTag(token)

    def endTagTableCell(self, token):
        if self.tree.elementInScope(token['name'], variant='table'):
            self.tree.generateImpliedEndTags(token['name'])
            if self.tree.openElements[-1].name != token['name']:
                self.parser.parseError('unexpected-cell-end-tag', {'name': token['name']})
                while True:
                    node = self.tree.openElements.pop()
                    if node.name == token['name']:
                        break
            else:
                self.tree.openElements.pop()
            self.tree.clearActiveFormattingElements()
            self.parser.phase = self.parser.phases['inRow']
        else:
            self.parser.parseError('unexpected-end-tag', {'name': token['name']})

    def endTagIgnore(self, token):
        self.parser.parseError('unexpected-end-tag', {'name': token['name']})

    def endTagImply(self, token):
        if self.tree.elementInScope(token['name'], variant='table'):
            self.closeCell()
            return token
        else:
            self.parser.parseError()

    def endTagOther(self, token):
        return self.parser.phases['inBody'].processEndTag(token)
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), (('caption', 'col', 'colgroup', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), startTagTableOther)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([(('td', 'th'), endTagTableCell), (('body', 'caption', 'col', 'colgroup', 'html'), endTagIgnore), (('table', 'tbody', 'tfoot', 'thead', 'tr'), endTagImply)])
    endTagHandler.default = endTagOther