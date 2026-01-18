from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
def __processPlaceholders(self, data: str | None, parent: etree.Element, isText: bool=True) -> list[tuple[etree.Element, list[str]]]:
    """
        Process string with placeholders and generate `ElementTree` tree.

        Arguments:
            data: String with placeholders instead of `ElementTree` elements.
            parent: Element, which contains processing inline data.
            isText: Boolean variable, True - it's text, False - it's a tail.

        Returns:
            List with `ElementTree` elements with applied inline patterns.

        """

    def linkText(text: str | None) -> None:
        if text:
            if result:
                if result[-1][0].tail:
                    result[-1][0].tail += text
                else:
                    result[-1][0].tail = text
            elif not isText:
                if parent.tail:
                    parent.tail += text
                else:
                    parent.tail = text
            elif parent.text:
                parent.text += text
            else:
                parent.text = text
    result = []
    strartIndex = 0
    while data:
        index = data.find(self.__placeholder_prefix, strartIndex)
        if index != -1:
            id, phEndIndex = self.__findPlaceholder(data, index)
            if id in self.stashed_nodes:
                node = self.stashed_nodes.get(id)
                if index > 0:
                    text = data[strartIndex:index]
                    linkText(text)
                if not isinstance(node, str):
                    for child in [node] + list(node):
                        if child.tail:
                            if child.tail.strip():
                                self.__processElementText(node, child, False)
                        if child.text:
                            if child.text.strip():
                                self.__processElementText(child, child)
                else:
                    linkText(node)
                    strartIndex = phEndIndex
                    continue
                strartIndex = phEndIndex
                result.append((node, self.ancestors[:]))
            else:
                end = index + len(self.__placeholder_prefix)
                linkText(data[strartIndex:end])
                strartIndex = end
        else:
            text = data[strartIndex:]
            if isinstance(data, util.AtomicString):
                text = util.AtomicString(text)
            linkText(text)
            data = ''
    return result