from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
class InBodyPhase(Phase):
    __slots__ = ('processSpaceCharacters',)

    def __init__(self, *args, **kwargs):
        super(InBodyPhase, self).__init__(*args, **kwargs)
        self.processSpaceCharacters = self.processSpaceCharactersNonPre

    def isMatchingFormattingElement(self, node1, node2):
        return node1.name == node2.name and node1.namespace == node2.namespace and (node1.attributes == node2.attributes)

    def addFormattingElement(self, token):
        self.tree.insertElement(token)
        element = self.tree.openElements[-1]
        matchingElements = []
        for node in self.tree.activeFormattingElements[::-1]:
            if node is Marker:
                break
            elif self.isMatchingFormattingElement(node, element):
                matchingElements.append(node)
        assert len(matchingElements) <= 3
        if len(matchingElements) == 3:
            self.tree.activeFormattingElements.remove(matchingElements[-1])
        self.tree.activeFormattingElements.append(element)

    def processEOF(self):
        allowed_elements = frozenset(('dd', 'dt', 'li', 'p', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'body', 'html'))
        for node in self.tree.openElements[::-1]:
            if node.name not in allowed_elements:
                self.parser.parseError('expected-closing-tag-but-got-eof')
                break

    def processSpaceCharactersDropNewline(self, token):
        data = token['data']
        self.processSpaceCharacters = self.processSpaceCharactersNonPre
        if data.startswith('\n') and self.tree.openElements[-1].name in ('pre', 'listing', 'textarea') and (not self.tree.openElements[-1].hasContent()):
            data = data[1:]
        if data:
            self.tree.reconstructActiveFormattingElements()
            self.tree.insertText(data)

    def processCharacters(self, token):
        if token['data'] == '\x00':
            return
        self.tree.reconstructActiveFormattingElements()
        self.tree.insertText(token['data'])
        if self.parser.framesetOK and any([char not in spaceCharacters for char in token['data']]):
            self.parser.framesetOK = False

    def processSpaceCharactersNonPre(self, token):
        self.tree.reconstructActiveFormattingElements()
        self.tree.insertText(token['data'])

    def startTagProcessInHead(self, token):
        return self.parser.phases['inHead'].processStartTag(token)

    def startTagBody(self, token):
        self.parser.parseError('unexpected-start-tag', {'name': 'body'})
        if len(self.tree.openElements) == 1 or self.tree.openElements[1].name != 'body':
            assert self.parser.innerHTML
        else:
            self.parser.framesetOK = False
            for attr, value in token['data'].items():
                if attr not in self.tree.openElements[1].attributes:
                    self.tree.openElements[1].attributes[attr] = value

    def startTagFrameset(self, token):
        self.parser.parseError('unexpected-start-tag', {'name': 'frameset'})
        if len(self.tree.openElements) == 1 or self.tree.openElements[1].name != 'body':
            assert self.parser.innerHTML
        elif not self.parser.framesetOK:
            pass
        else:
            if self.tree.openElements[1].parent:
                self.tree.openElements[1].parent.removeChild(self.tree.openElements[1])
            while self.tree.openElements[-1].name != 'html':
                self.tree.openElements.pop()
            self.tree.insertElement(token)
            self.parser.phase = self.parser.phases['inFrameset']

    def startTagCloseP(self, token):
        if self.tree.elementInScope('p', variant='button'):
            self.endTagP(impliedTagToken('p'))
        self.tree.insertElement(token)

    def startTagPreListing(self, token):
        if self.tree.elementInScope('p', variant='button'):
            self.endTagP(impliedTagToken('p'))
        self.tree.insertElement(token)
        self.parser.framesetOK = False
        self.processSpaceCharacters = self.processSpaceCharactersDropNewline

    def startTagForm(self, token):
        if self.tree.formPointer:
            self.parser.parseError('unexpected-start-tag', {'name': 'form'})
        else:
            if self.tree.elementInScope('p', variant='button'):
                self.endTagP(impliedTagToken('p'))
            self.tree.insertElement(token)
            self.tree.formPointer = self.tree.openElements[-1]

    def startTagListItem(self, token):
        self.parser.framesetOK = False
        stopNamesMap = {'li': ['li'], 'dt': ['dt', 'dd'], 'dd': ['dt', 'dd']}
        stopNames = stopNamesMap[token['name']]
        for node in reversed(self.tree.openElements):
            if node.name in stopNames:
                self.parser.phase.processEndTag(impliedTagToken(node.name, 'EndTag'))
                break
            if node.nameTuple in specialElements and node.name not in ('address', 'div', 'p'):
                break
        if self.tree.elementInScope('p', variant='button'):
            self.parser.phase.processEndTag(impliedTagToken('p', 'EndTag'))
        self.tree.insertElement(token)

    def startTagPlaintext(self, token):
        if self.tree.elementInScope('p', variant='button'):
            self.endTagP(impliedTagToken('p'))
        self.tree.insertElement(token)
        self.parser.tokenizer.state = self.parser.tokenizer.plaintextState

    def startTagHeading(self, token):
        if self.tree.elementInScope('p', variant='button'):
            self.endTagP(impliedTagToken('p'))
        if self.tree.openElements[-1].name in headingElements:
            self.parser.parseError('unexpected-start-tag', {'name': token['name']})
            self.tree.openElements.pop()
        self.tree.insertElement(token)

    def startTagA(self, token):
        afeAElement = self.tree.elementInActiveFormattingElements('a')
        if afeAElement:
            self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'a', 'endName': 'a'})
            self.endTagFormatting(impliedTagToken('a'))
            if afeAElement in self.tree.openElements:
                self.tree.openElements.remove(afeAElement)
            if afeAElement in self.tree.activeFormattingElements:
                self.tree.activeFormattingElements.remove(afeAElement)
        self.tree.reconstructActiveFormattingElements()
        self.addFormattingElement(token)

    def startTagFormatting(self, token):
        self.tree.reconstructActiveFormattingElements()
        self.addFormattingElement(token)

    def startTagNobr(self, token):
        self.tree.reconstructActiveFormattingElements()
        if self.tree.elementInScope('nobr'):
            self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'nobr', 'endName': 'nobr'})
            self.processEndTag(impliedTagToken('nobr'))
            self.tree.reconstructActiveFormattingElements()
        self.addFormattingElement(token)

    def startTagButton(self, token):
        if self.tree.elementInScope('button'):
            self.parser.parseError('unexpected-start-tag-implies-end-tag', {'startName': 'button', 'endName': 'button'})
            self.processEndTag(impliedTagToken('button'))
            return token
        else:
            self.tree.reconstructActiveFormattingElements()
            self.tree.insertElement(token)
            self.parser.framesetOK = False

    def startTagAppletMarqueeObject(self, token):
        self.tree.reconstructActiveFormattingElements()
        self.tree.insertElement(token)
        self.tree.activeFormattingElements.append(Marker)
        self.parser.framesetOK = False

    def startTagXmp(self, token):
        if self.tree.elementInScope('p', variant='button'):
            self.endTagP(impliedTagToken('p'))
        self.tree.reconstructActiveFormattingElements()
        self.parser.framesetOK = False
        self.parser.parseRCDataRawtext(token, 'RAWTEXT')

    def startTagTable(self, token):
        if self.parser.compatMode != 'quirks':
            if self.tree.elementInScope('p', variant='button'):
                self.processEndTag(impliedTagToken('p'))
        self.tree.insertElement(token)
        self.parser.framesetOK = False
        self.parser.phase = self.parser.phases['inTable']

    def startTagVoidFormatting(self, token):
        self.tree.reconstructActiveFormattingElements()
        self.tree.insertElement(token)
        self.tree.openElements.pop()
        token['selfClosingAcknowledged'] = True
        self.parser.framesetOK = False

    def startTagInput(self, token):
        framesetOK = self.parser.framesetOK
        self.startTagVoidFormatting(token)
        if 'type' in token['data'] and token['data']['type'].translate(asciiUpper2Lower) == 'hidden':
            self.parser.framesetOK = framesetOK

    def startTagParamSource(self, token):
        self.tree.insertElement(token)
        self.tree.openElements.pop()
        token['selfClosingAcknowledged'] = True

    def startTagHr(self, token):
        if self.tree.elementInScope('p', variant='button'):
            self.endTagP(impliedTagToken('p'))
        self.tree.insertElement(token)
        self.tree.openElements.pop()
        token['selfClosingAcknowledged'] = True
        self.parser.framesetOK = False

    def startTagImage(self, token):
        self.parser.parseError('unexpected-start-tag-treated-as', {'originalName': 'image', 'newName': 'img'})
        self.processStartTag(impliedTagToken('img', 'StartTag', attributes=token['data'], selfClosing=token['selfClosing']))

    def startTagIsIndex(self, token):
        self.parser.parseError('deprecated-tag', {'name': 'isindex'})
        if self.tree.formPointer:
            return
        form_attrs = {}
        if 'action' in token['data']:
            form_attrs['action'] = token['data']['action']
        self.processStartTag(impliedTagToken('form', 'StartTag', attributes=form_attrs))
        self.processStartTag(impliedTagToken('hr', 'StartTag'))
        self.processStartTag(impliedTagToken('label', 'StartTag'))
        if 'prompt' in token['data']:
            prompt = token['data']['prompt']
        else:
            prompt = 'This is a searchable index. Enter search keywords: '
        self.processCharacters({'type': tokenTypes['Characters'], 'data': prompt})
        attributes = token['data'].copy()
        if 'action' in attributes:
            del attributes['action']
        if 'prompt' in attributes:
            del attributes['prompt']
        attributes['name'] = 'isindex'
        self.processStartTag(impliedTagToken('input', 'StartTag', attributes=attributes, selfClosing=token['selfClosing']))
        self.processEndTag(impliedTagToken('label'))
        self.processStartTag(impliedTagToken('hr', 'StartTag'))
        self.processEndTag(impliedTagToken('form'))

    def startTagTextarea(self, token):
        self.tree.insertElement(token)
        self.parser.tokenizer.state = self.parser.tokenizer.rcdataState
        self.processSpaceCharacters = self.processSpaceCharactersDropNewline
        self.parser.framesetOK = False

    def startTagIFrame(self, token):
        self.parser.framesetOK = False
        self.startTagRawtext(token)

    def startTagNoscript(self, token):
        if self.parser.scripting:
            self.startTagRawtext(token)
        else:
            self.startTagOther(token)

    def startTagRawtext(self, token):
        """iframe, noembed noframes, noscript(if scripting enabled)"""
        self.parser.parseRCDataRawtext(token, 'RAWTEXT')

    def startTagOpt(self, token):
        if self.tree.openElements[-1].name == 'option':
            self.parser.phase.processEndTag(impliedTagToken('option'))
        self.tree.reconstructActiveFormattingElements()
        self.parser.tree.insertElement(token)

    def startTagSelect(self, token):
        self.tree.reconstructActiveFormattingElements()
        self.tree.insertElement(token)
        self.parser.framesetOK = False
        if self.parser.phase in (self.parser.phases['inTable'], self.parser.phases['inCaption'], self.parser.phases['inColumnGroup'], self.parser.phases['inTableBody'], self.parser.phases['inRow'], self.parser.phases['inCell']):
            self.parser.phase = self.parser.phases['inSelectInTable']
        else:
            self.parser.phase = self.parser.phases['inSelect']

    def startTagRpRt(self, token):
        if self.tree.elementInScope('ruby'):
            self.tree.generateImpliedEndTags()
            if self.tree.openElements[-1].name != 'ruby':
                self.parser.parseError()
        self.tree.insertElement(token)

    def startTagMath(self, token):
        self.tree.reconstructActiveFormattingElements()
        self.parser.adjustMathMLAttributes(token)
        self.parser.adjustForeignAttributes(token)
        token['namespace'] = namespaces['mathml']
        self.tree.insertElement(token)
        if token['selfClosing']:
            self.tree.openElements.pop()
            token['selfClosingAcknowledged'] = True

    def startTagSvg(self, token):
        self.tree.reconstructActiveFormattingElements()
        self.parser.adjustSVGAttributes(token)
        self.parser.adjustForeignAttributes(token)
        token['namespace'] = namespaces['svg']
        self.tree.insertElement(token)
        if token['selfClosing']:
            self.tree.openElements.pop()
            token['selfClosingAcknowledged'] = True

    def startTagMisplaced(self, token):
        """ Elements that should be children of other elements that have a
            different insertion mode; here they are ignored
            "caption", "col", "colgroup", "frame", "frameset", "head",
            "option", "optgroup", "tbody", "td", "tfoot", "th", "thead",
            "tr", "noscript"
            """
        self.parser.parseError('unexpected-start-tag-ignored', {'name': token['name']})

    def startTagOther(self, token):
        self.tree.reconstructActiveFormattingElements()
        self.tree.insertElement(token)

    def endTagP(self, token):
        if not self.tree.elementInScope('p', variant='button'):
            self.startTagCloseP(impliedTagToken('p', 'StartTag'))
            self.parser.parseError('unexpected-end-tag', {'name': 'p'})
            self.endTagP(impliedTagToken('p', 'EndTag'))
        else:
            self.tree.generateImpliedEndTags('p')
            if self.tree.openElements[-1].name != 'p':
                self.parser.parseError('unexpected-end-tag', {'name': 'p'})
            node = self.tree.openElements.pop()
            while node.name != 'p':
                node = self.tree.openElements.pop()

    def endTagBody(self, token):
        if not self.tree.elementInScope('body'):
            self.parser.parseError()
            return
        elif self.tree.openElements[-1].name != 'body':
            for node in self.tree.openElements[2:]:
                if node.name not in frozenset(('dd', 'dt', 'li', 'optgroup', 'option', 'p', 'rp', 'rt', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'body', 'html')):
                    self.parser.parseError('expected-one-end-tag-but-got-another', {'gotName': 'body', 'expectedName': node.name})
                    break
        self.parser.phase = self.parser.phases['afterBody']

    def endTagHtml(self, token):
        if self.tree.elementInScope('body'):
            self.endTagBody(impliedTagToken('body'))
            return token

    def endTagBlock(self, token):
        if token['name'] == 'pre':
            self.processSpaceCharacters = self.processSpaceCharactersNonPre
        inScope = self.tree.elementInScope(token['name'])
        if inScope:
            self.tree.generateImpliedEndTags()
        if self.tree.openElements[-1].name != token['name']:
            self.parser.parseError('end-tag-too-early', {'name': token['name']})
        if inScope:
            node = self.tree.openElements.pop()
            while node.name != token['name']:
                node = self.tree.openElements.pop()

    def endTagForm(self, token):
        node = self.tree.formPointer
        self.tree.formPointer = None
        if node is None or not self.tree.elementInScope(node):
            self.parser.parseError('unexpected-end-tag', {'name': 'form'})
        else:
            self.tree.generateImpliedEndTags()
            if self.tree.openElements[-1] != node:
                self.parser.parseError('end-tag-too-early-ignored', {'name': 'form'})
            self.tree.openElements.remove(node)

    def endTagListItem(self, token):
        if token['name'] == 'li':
            variant = 'list'
        else:
            variant = None
        if not self.tree.elementInScope(token['name'], variant=variant):
            self.parser.parseError('unexpected-end-tag', {'name': token['name']})
        else:
            self.tree.generateImpliedEndTags(exclude=token['name'])
            if self.tree.openElements[-1].name != token['name']:
                self.parser.parseError('end-tag-too-early', {'name': token['name']})
            node = self.tree.openElements.pop()
            while node.name != token['name']:
                node = self.tree.openElements.pop()

    def endTagHeading(self, token):
        for item in headingElements:
            if self.tree.elementInScope(item):
                self.tree.generateImpliedEndTags()
                break
        if self.tree.openElements[-1].name != token['name']:
            self.parser.parseError('end-tag-too-early', {'name': token['name']})
        for item in headingElements:
            if self.tree.elementInScope(item):
                item = self.tree.openElements.pop()
                while item.name not in headingElements:
                    item = self.tree.openElements.pop()
                break

    def endTagFormatting(self, token):
        """The much-feared adoption agency algorithm"""
        outerLoopCounter = 0
        while outerLoopCounter < 8:
            outerLoopCounter += 1
            formattingElement = self.tree.elementInActiveFormattingElements(token['name'])
            if not formattingElement or (formattingElement in self.tree.openElements and (not self.tree.elementInScope(formattingElement.name))):
                self.endTagOther(token)
                return
            elif formattingElement not in self.tree.openElements:
                self.parser.parseError('adoption-agency-1.2', {'name': token['name']})
                self.tree.activeFormattingElements.remove(formattingElement)
                return
            elif not self.tree.elementInScope(formattingElement.name):
                self.parser.parseError('adoption-agency-4.4', {'name': token['name']})
                return
            elif formattingElement != self.tree.openElements[-1]:
                self.parser.parseError('adoption-agency-1.3', {'name': token['name']})
            afeIndex = self.tree.openElements.index(formattingElement)
            furthestBlock = None
            for element in self.tree.openElements[afeIndex:]:
                if element.nameTuple in specialElements:
                    furthestBlock = element
                    break
            if furthestBlock is None:
                element = self.tree.openElements.pop()
                while element != formattingElement:
                    element = self.tree.openElements.pop()
                self.tree.activeFormattingElements.remove(element)
                return
            commonAncestor = self.tree.openElements[afeIndex - 1]
            bookmark = self.tree.activeFormattingElements.index(formattingElement)
            lastNode = node = furthestBlock
            innerLoopCounter = 0
            index = self.tree.openElements.index(node)
            while innerLoopCounter < 3:
                innerLoopCounter += 1
                index -= 1
                node = self.tree.openElements[index]
                if node not in self.tree.activeFormattingElements:
                    self.tree.openElements.remove(node)
                    continue
                if node == formattingElement:
                    break
                if lastNode == furthestBlock:
                    bookmark = self.tree.activeFormattingElements.index(node) + 1
                clone = node.cloneNode()
                self.tree.activeFormattingElements[self.tree.activeFormattingElements.index(node)] = clone
                self.tree.openElements[self.tree.openElements.index(node)] = clone
                node = clone
                if lastNode.parent:
                    lastNode.parent.removeChild(lastNode)
                node.appendChild(lastNode)
                lastNode = node
            if lastNode.parent:
                lastNode.parent.removeChild(lastNode)
            if commonAncestor.name in frozenset(('table', 'tbody', 'tfoot', 'thead', 'tr')):
                parent, insertBefore = self.tree.getTableMisnestedNodePosition()
                parent.insertBefore(lastNode, insertBefore)
            else:
                commonAncestor.appendChild(lastNode)
            clone = formattingElement.cloneNode()
            furthestBlock.reparentChildren(clone)
            furthestBlock.appendChild(clone)
            self.tree.activeFormattingElements.remove(formattingElement)
            self.tree.activeFormattingElements.insert(bookmark, clone)
            self.tree.openElements.remove(formattingElement)
            self.tree.openElements.insert(self.tree.openElements.index(furthestBlock) + 1, clone)

    def endTagAppletMarqueeObject(self, token):
        if self.tree.elementInScope(token['name']):
            self.tree.generateImpliedEndTags()
        if self.tree.openElements[-1].name != token['name']:
            self.parser.parseError('end-tag-too-early', {'name': token['name']})
        if self.tree.elementInScope(token['name']):
            element = self.tree.openElements.pop()
            while element.name != token['name']:
                element = self.tree.openElements.pop()
            self.tree.clearActiveFormattingElements()

    def endTagBr(self, token):
        self.parser.parseError('unexpected-end-tag-treated-as', {'originalName': 'br', 'newName': 'br element'})
        self.tree.reconstructActiveFormattingElements()
        self.tree.insertElement(impliedTagToken('br', 'StartTag'))
        self.tree.openElements.pop()

    def endTagOther(self, token):
        for node in self.tree.openElements[::-1]:
            if node.name == token['name']:
                self.tree.generateImpliedEndTags(exclude=token['name'])
                if self.tree.openElements[-1].name != token['name']:
                    self.parser.parseError('unexpected-end-tag', {'name': token['name']})
                while self.tree.openElements.pop() != node:
                    pass
                break
            elif node.nameTuple in specialElements:
                self.parser.parseError('unexpected-end-tag', {'name': token['name']})
                break
    startTagHandler = _utils.MethodDispatcher([('html', Phase.startTagHtml), (('base', 'basefont', 'bgsound', 'command', 'link', 'meta', 'script', 'style', 'title'), startTagProcessInHead), ('body', startTagBody), ('frameset', startTagFrameset), (('address', 'article', 'aside', 'blockquote', 'center', 'details', 'dir', 'div', 'dl', 'fieldset', 'figcaption', 'figure', 'footer', 'header', 'hgroup', 'main', 'menu', 'nav', 'ol', 'p', 'section', 'summary', 'ul'), startTagCloseP), (headingElements, startTagHeading), (('pre', 'listing'), startTagPreListing), ('form', startTagForm), (('li', 'dd', 'dt'), startTagListItem), ('plaintext', startTagPlaintext), ('a', startTagA), (('b', 'big', 'code', 'em', 'font', 'i', 's', 'small', 'strike', 'strong', 'tt', 'u'), startTagFormatting), ('nobr', startTagNobr), ('button', startTagButton), (('applet', 'marquee', 'object'), startTagAppletMarqueeObject), ('xmp', startTagXmp), ('table', startTagTable), (('area', 'br', 'embed', 'img', 'keygen', 'wbr'), startTagVoidFormatting), (('param', 'source', 'track'), startTagParamSource), ('input', startTagInput), ('hr', startTagHr), ('image', startTagImage), ('isindex', startTagIsIndex), ('textarea', startTagTextarea), ('iframe', startTagIFrame), ('noscript', startTagNoscript), (('noembed', 'noframes'), startTagRawtext), ('select', startTagSelect), (('rp', 'rt'), startTagRpRt), (('option', 'optgroup'), startTagOpt), ('math', startTagMath), ('svg', startTagSvg), (('caption', 'col', 'colgroup', 'frame', 'head', 'tbody', 'td', 'tfoot', 'th', 'thead', 'tr'), startTagMisplaced)])
    startTagHandler.default = startTagOther
    endTagHandler = _utils.MethodDispatcher([('body', endTagBody), ('html', endTagHtml), (('address', 'article', 'aside', 'blockquote', 'button', 'center', 'details', 'dialog', 'dir', 'div', 'dl', 'fieldset', 'figcaption', 'figure', 'footer', 'header', 'hgroup', 'listing', 'main', 'menu', 'nav', 'ol', 'pre', 'section', 'summary', 'ul'), endTagBlock), ('form', endTagForm), ('p', endTagP), (('dd', 'dt', 'li'), endTagListItem), (headingElements, endTagHeading), (('a', 'b', 'big', 'code', 'em', 'font', 'i', 'nobr', 's', 'small', 'strike', 'strong', 'tt', 'u'), endTagFormatting), (('applet', 'marquee', 'object'), endTagAppletMarqueeObject), ('br', endTagBr)])
    endTagHandler.default = endTagOther