from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InHeadPhase(Phase):
    __slots__ = tuple()

    def processEOF(self):
        self.anythingElse()
        return True

    def processCharacters(self, token):
        self.anythingElse()
        return token

    def startTagHtml(self, token):
        return self.parser.phases['inBody'].processStartTag(token)

    def startTagHead(self, token):
        self.parser.parseError('two-heads-are-not-better-than-one')

    def startTagBaseLinkCommand(self, token):
        self.tree.insertElement(token)
        self.tree.openElements.pop()
        token['selfClosingAcknowledged'] = True

    def startTagMeta(self, token):
        self.tree.insertElement(token)
        self.tree.openElements.pop()
        token['selfClosingAcknowledged'] = True
        attributes = token['data']
        if self.parser.tokenizer.stream.charEncoding[1] == 'tentative':
            if 'charset' in attributes:
                self.parser.tokenizer.stream.changeEncoding(attributes['charset'])
            elif 'content' in attributes and 'http-equiv' in attributes and (attributes['http-equiv'].lower() == 'content-type'):
                data = _inputstream.EncodingBytes(attributes['content'].encode('utf-8'))
                parser = _inputstream.ContentAttrParser(data)
                codec = parser.parse()
                self.parser.tokenizer.stream.changeEncoding(codec)

    def startTagTitle(self, token):
        self.parser.parseRCDataRawtext(token, 'RCDATA')

    def startTagNoFramesStyle(self, token):
        self.parser.parseRCDataRawtext(token, 'RAWTEXT')

    def startTagNoscript(self, token):
        if self.parser.scripting:
            self.parser.parseRCDataRawtext(token, 'RAWTEXT')
        else:
            self.tree.insertElement(token)
            self.parser.phase = self.parser.phases['inHeadNoscript']

    def startTagScript(self, token):
        self.tree.insertElement(token)
        self.parser.tokenizer.state = self.parser.tokenizer.scriptDataState
        self.parser.originalPhase = self.parser.phase
        self.parser.phase = self.parser.phases['text']

    def startTagOther(self, token):
        self.anythingElse()
        return token

    def endTagHead(self, token):
        node = self.parser.tree.openElements.pop()
        assert node.name == 'head', 'Expected head got %s' % node.name
        self.parser.phase = self.parser.phases['afterHead']

    def endTagHtmlBodyBr(self, token):
        self.anythingElse()
        return token

    def endTagOther(self, token):
        self.parser.parseError('unexpected-end-tag', {'name': token['name']})

    def anythingElse(self):
        self.endTagHead(impliedTagToken('head'))
    startTagHandler = _utils.MethodDispatcher([('html', startTagHtml), ('title', startTagTitle), (('noframes', 'style'), startTagNoFramesStyle), ('noscript', startTagNoscript), ('script', startTagScript), (('base', 'basefont', 'bgsound', 'command', 'link'), startTagBaseLinkCommand), ('meta', startTagMeta), ('head', startTagHead)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('head', endTagHead), (('br', 'html', 'body'), endTagHtmlBodyBr)])
    endTagHandler.default = endTagOther