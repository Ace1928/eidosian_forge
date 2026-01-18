from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InTableTextPhase(Phase):
    __slots__ = ('originalPhase', 'characterTokens')

    def __init__(self, *args, **kwargs):
        super(InTableTextPhase, self).__init__(*args, **kwargs)
        self.originalPhase = None
        self.characterTokens = []

    def flushCharacters(self):
        data = ''.join([item['data'] for item in self.characterTokens])
        if any([item not in spaceCharacters for item in data]):
            token = {'type': tokenTypes['Characters'], 'data': data}
            self.parser.phases['inTable'].insertText(token)
        elif data:
            self.tree.insertText(data)
        self.characterTokens = []

    def processComment(self, token):
        self.flushCharacters()
        self.parser.phase = self.originalPhase
        return token

    def processEOF(self):
        self.flushCharacters()
        self.parser.phase = self.originalPhase
        return True

    def processCharacters(self, token):
        if token['data'] == '\x00':
            return
        self.characterTokens.append(token)

    def processSpaceCharacters(self, token):
        self.characterTokens.append(token)

    def processStartTag(self, token):
        self.flushCharacters()
        self.parser.phase = self.originalPhase
        return token

    def processEndTag(self, token):
        self.flushCharacters()
        self.parser.phase = self.originalPhase
        return token