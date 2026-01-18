from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InSelectPhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        if self.tree.openElements[-1].name != 'html':
            self.parser.parseError('eof-in-select')
        else:
            assert self.parser.innerHTML

    def processCharacters(self, token):
        if token['data'] == '\x00':
            return
        self.tree.insertText(token['data'])

    def startTagOption(self, token):
        if self.tree.openElements[-1].name == 'option':
            self.tree.openElements.pop()
        self.tree.insertElement(token)

    def startTagOptgroup(self, token):
        if self.tree.openElements[-1].name == 'option':
            self.tree.openElements.pop()
        if self.tree.openElements[-1].name == 'optgroup':
            self.tree.openElements.pop()
        self.tree.insertElement(token)

    def startTagSelect(self, token):
        self.parser.parseError('unexpected-select-in-select')
        self.endTagSelect(impliedTagToken('select'))

    def startTagInput(self, token):
        self.parser.parseError('unexpected-input-in-select')
        if self.tree.elementInScope('select', variant='select'):
            self.endTagSelect(impliedTagToken('select'))
            return token
        else:
            assert self.parser.innerHTML

    def startTagScript(self, token):
        return self.parser.phases['inHead'].processStartTag(token)

    def startTagOther(self, token):
        self.parser.parseError('unexpected-start-tag-in-select', {'name': token['name']})

    def endTagOption(self, token):
        if self.tree.openElements[-1].name == 'option':
            self.tree.openElements.pop()
        else:
            self.parser.parseError('unexpected-end-tag-in-select', {'name': 'option'})

    def endTagOptgroup(self, token):
        if self.tree.openElements[-1].name == 'option' and self.tree.openElements[-2].name == 'optgroup':
            self.tree.openElements.pop()
        if self.tree.openElements[-1].name == 'optgroup':
            self.tree.openElements.pop()
        else:
            self.parser.parseError('unexpected-end-tag-in-select', {'name': 'optgroup'})

    def endTagSelect(self, token):
        if self.tree.elementInScope('select', variant='select'):
            node = self.tree.openElements.pop()
            while node.name != 'select':
                node = self.tree.openElements.pop()
            self.parser.resetInsertionMode()
        else:
            assert self.parser.innerHTML
            self.parser.parseError()

    def endTagOther(self, token):
        self.parser.parseError('unexpected-end-tag-in-select', {'name': token['name']})
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), ('option', startTagOption), ('optgroup', startTagOptgroup), ('select', startTagSelect), (('input', 'keygen', 'textarea'), startTagInput), ('script', startTagScript)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('option', endTagOption), ('optgroup', endTagOptgroup), ('select', endTagSelect)])
    endTagHandler.default = endTagOther