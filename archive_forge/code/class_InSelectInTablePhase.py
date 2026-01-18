from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InSelectInTablePhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        self.parser.phases['inSelect'].processEOF()

    def processCharacters(self, token):
        return self.parser.phases['inSelect'].processCharacters(token)

    def startTagTable(self, token):
        self.parser.parseError('unexpected-table-element-start-tag-in-select-in-table', {'name': token['name']})
        self.endTagOther(impliedTagToken('select'))
        return token

    def startTagOther(self, token):
        return self.parser.phases['inSelect'].processStartTag(token)

    def endTagTable(self, token):
        self.parser.parseError('unexpected-table-element-end-tag-in-select-in-table', {'name': token['name']})
        if self.tree.elementInScope(token['name'], variant='table'):
            self.endTagOther(impliedTagToken('select'))
            return token

    def endTagOther(self, token):
        return self.parser.phases['inSelect'].processEndTag(token)
    startTagHandler = _utils.MethodDispatcher([(('caption', 'table', 'tbody', 'tfoot', 'thead', 'tr', 'td', 'th'), startTagTable)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([(('caption', 'table', 'tbody', 'tfoot', 'thead', 'tr', 'td', 'th'), endTagTable)])
    endTagHandler.default = endTagOther