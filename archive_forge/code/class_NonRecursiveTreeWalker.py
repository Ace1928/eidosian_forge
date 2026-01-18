from __future__ import absolute_import, division, unicode_literals
from xml.dom import Node
from ..constants import namespaces, voidElements, spaceCharacters
class NonRecursiveTreeWalker(TreeWalker):

    def getNodeDetails(self, node):
        raise NotImplementedError

    def getFirstChild(self, node):
        raise NotImplementedError

    def getNextSibling(self, node):
        raise NotImplementedError

    def getParentNode(self, node):
        raise NotImplementedError

    def __iter__(self):
        currentNode = self.tree
        while currentNode is not None:
            details = self.getNodeDetails(currentNode)
            type, details = (details[0], details[1:])
            hasChildren = False
            if type == DOCTYPE:
                yield self.doctype(*details)
            elif type == TEXT:
                for token in self.text(*details):
                    yield token
            elif type == ELEMENT:
                namespace, name, attributes, hasChildren = details
                if (not namespace or namespace == namespaces['html']) and name in voidElements:
                    for token in self.emptyTag(namespace, name, attributes, hasChildren):
                        yield token
                    hasChildren = False
                else:
                    yield self.startTag(namespace, name, attributes)
            elif type == COMMENT:
                yield self.comment(details[0])
            elif type == ENTITY:
                yield self.entity(details[0])
            elif type == DOCUMENT:
                hasChildren = True
            else:
                yield self.unknown(details[0])
            if hasChildren:
                firstChild = self.getFirstChild(currentNode)
            else:
                firstChild = None
            if firstChild is not None:
                currentNode = firstChild
            else:
                while currentNode is not None:
                    details = self.getNodeDetails(currentNode)
                    type, details = (details[0], details[1:])
                    if type == ELEMENT:
                        namespace, name, attributes, hasChildren = details
                        if namespace and namespace != namespaces['html'] or name not in voidElements:
                            yield self.endTag(namespace, name)
                    if self.tree is currentNode:
                        currentNode = None
                        break
                    nextSibling = self.getNextSibling(currentNode)
                    if nextSibling is not None:
                        currentNode = nextSibling
                        break
                    else:
                        currentNode = self.getParentNode(currentNode)