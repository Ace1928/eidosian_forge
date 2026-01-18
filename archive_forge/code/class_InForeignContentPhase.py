from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InForeignContentPhase(Phase):
    __slots__ = tuple()
    breakoutElements = frozenset(['b', 'big', 'blockquote', 'body', 'br', 'center', 'code', 'dd', 'div', 'dl', 'dt', 'em', 'embed', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'hr', 'i', 'img', 'li', 'listing', 'menu', 'meta', 'nobr', 'ol', 'p', 'pre', 'ruby', 's', 'small', 'span', 'strong', 'strike', 'sub', 'sup', 'table', 'tt', 'u', 'ul', 'var'])

    def adjustSVGTagNames(self, token):
        replacements = {'altglyph': 'altGlyph', 'altglyphdef': 'altGlyphDef', 'altglyphitem': 'altGlyphItem', 'animatecolor': 'animateColor', 'animatemotion': 'animateMotion', 'animatetransform': 'animateTransform', 'clippath': 'clipPath', 'feblend': 'feBlend', 'fecolormatrix': 'feColorMatrix', 'fecomponenttransfer': 'feComponentTransfer', 'fecomposite': 'feComposite', 'feconvolvematrix': 'feConvolveMatrix', 'fediffuselighting': 'feDiffuseLighting', 'fedisplacementmap': 'feDisplacementMap', 'fedistantlight': 'feDistantLight', 'feflood': 'feFlood', 'fefunca': 'feFuncA', 'fefuncb': 'feFuncB', 'fefuncg': 'feFuncG', 'fefuncr': 'feFuncR', 'fegaussianblur': 'feGaussianBlur', 'feimage': 'feImage', 'femerge': 'feMerge', 'femergenode': 'feMergeNode', 'femorphology': 'feMorphology', 'feoffset': 'feOffset', 'fepointlight': 'fePointLight', 'fespecularlighting': 'feSpecularLighting', 'fespotlight': 'feSpotLight', 'fetile': 'feTile', 'feturbulence': 'feTurbulence', 'foreignobject': 'foreignObject', 'glyphref': 'glyphRef', 'lineargradient': 'linearGradient', 'radialgradient': 'radialGradient', 'textpath': 'textPath'}
        if token['name'] in replacements:
            token['name'] = replacements[token['name']]

    def processCharacters(self, token):
        if token['data'] == '\x00':
            token['data'] = '�'
        elif self.parser.framesetOK and any((char not in spaceCharacters for char in token['data'])):
            self.parser.framesetOK = False
        Phase.processCharacters(self, token)

    def processStartTag(self, token):
        currentNode = self.tree.openElements[-1]
        if token['name'] in self.breakoutElements or (token['name'] == 'font' and set(token['data'].keys()) & {'color', 'face', 'size'}):
            self.parser.parseError('unexpected-html-element-in-foreign-content', {'name': token['name']})
            while self.tree.openElements[-1].namespace != self.tree.defaultNamespace and (not self.parser.isHTMLIntegrationPoint(self.tree.openElements[-1])) and (not self.parser.isMathMLTextIntegrationPoint(self.tree.openElements[-1])):
                self.tree.openElements.pop()
            return token
        else:
            if currentNode.namespace == namespaces['mathml']:
                self.parser.adjustMathMLAttributes(token)
            elif currentNode.namespace == namespaces['svg']:
                self.adjustSVGTagNames(token)
                self.parser.adjustSVGAttributes(token)
            self.parser.adjustForeignAttributes(token)
            token['namespace'] = currentNode.namespace
            self.tree.insertElement(token)
            if token['selfClosing']:
                self.tree.openElements.pop()
                token['selfClosingAcknowledged'] = True

    def processEndTag(self, token):
        nodeIndex = len(self.tree.openElements) - 1
        node = self.tree.openElements[-1]
        if node.name.translate(asciiUpper2Lower) != token['name']:
            self.parser.parseError('unexpected-end-tag', {'name': token['name']})
        while True:
            if node.name.translate(asciiUpper2Lower) == token['name']:
                if self.parser.phase == self.parser.phases['inTableText']:
                    self.parser.phase.flushCharacters()
                    self.parser.phase = self.parser.phase.originalPhase
                while self.tree.openElements.pop() != node:
                    assert self.tree.openElements
                new_token = None
                break
            nodeIndex -= 1
            node = self.tree.openElements[nodeIndex]
            if node.namespace != self.tree.defaultNamespace:
                continue
            else:
                new_token = self.parser.phase.processEndTag(token)
                break
        return new_token