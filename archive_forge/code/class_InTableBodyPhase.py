from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InTableBodyPhase(Phase):
    __slots__ = tuple()

    def clearStackToTableBodyContext(self):
        while self.tree.openElements[-1].name not in ('tbody', 'tfoot', 'thead', 'html'):
            self.tree.openElements.pop()
        if self.tree.openElements[-1].name == 'html':
            assert self.parser.innerHTML

    def processEOF(self):
        self.parser.phases['inTable'].processEOF()

    def processSpaceCharacters(self, token):
        return self.parser.phases['inTable'].processSpaceCharacters(token)

    def processCharacters(self, token):
        return self.parser.phases['inTable'].processCharacters(token)

    def startTagTr(self, token):
        self.clearStackToTableBodyContext()
        self.tree.insertElement(token)
        self.parser.phase = self.parser.phases['inRow']

    def startTagTableCell(self, token):
        self.parser.parseError('unexpected-cell-in-table-body', {'name': token['name']})
        self.startTagTr(impliedTagToken('tr', 'StartTag'))
        return token

    def startTagTableOther(self, token):
        if self.tree.elementInScope('tbody', variant='table') or self.tree.elementInScope('thead', variant='table') or self.tree.elementInScope('tfoot', variant='table'):
            self.clearStackToTableBodyContext()
            self.endTagTableRowGroup(impliedTagToken(self.tree.openElements[-1].name))
            return token
        else:
            assert self.parser.innerHTML
            self.parser.parseError()

    def startTagOther(self, token):
        return self.parser.phases['inTable'].processStartTag(token)

    def endTagTableRowGroup(self, token):
        if self.tree.elementInScope(token['name'], variant='table'):
            self.clearStackToTableBodyContext()
            self.tree.openElements.pop()
            self.parser.phase = self.parser.phases['inTable']
        else:
            self.parser.parseError('unexpected-end-tag-in-table-body', {'name': token['name']})

    def endTagTable(self, token):
        if self.tree.elementInScope('tbody', variant='table') or self.tree.elementInScope('thead', variant='table') or self.tree.elementInScope('tfoot', variant='table'):
            self.clearStackToTableBodyContext()
            self.endTagTableRowGroup(impliedTagToken(self.tree.openElements[-1].name))
            return token
        else:
            assert self.parser.innerHTML
            self.parser.parseError()

    def endTagIgnore(self, token):
        self.parser.parseError('unexpected-end-tag-in-table-body', {'name': token['name']})

    def endTagOther(self, token):
        return self.parser.phases['inTable'].processEndTag(token)
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), ('tr', startTagTr), (('td', 'th'), startTagTableCell), (('caption', 'col', 'colgroup', 'tbody', 'tfoot', 'thead'), startTagTableOther)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([(('tbody', 'tfoot', 'thead'), endTagTableRowGroup), ('table', endTagTable), (('body', 'caption', 'col', 'colgroup', 'html', 'td', 'th', 'tr'), endTagIgnore)])
    endTagHandler.default = endTagOther