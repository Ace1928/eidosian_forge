from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document
class LazyElementNode(ElementNode):
    """
    A fully lazy element node, slower but better if the node does not
    to be used in a document context. The node extends descendants but
    does not record positions and a map of elements.
    """
    __slots__ = ()

    def __iter__(self) -> Iterator[ChildNodeType]:
        if not self.children:
            if self.elem.text is not None:
                self.children.append(TextNode(self.elem.text, self))
            if len(self.elem):
                for elem in self.elem:
                    if not callable(elem.tag):
                        nsmap = cast(Dict[Any, str], getattr(elem, 'nsmap', self.nsmap))
                        self.children.append(LazyElementNode(elem, self, nsmap=nsmap))
                    elif elem.tag.__name__ == 'Comment':
                        self.children.append(CommentNode(elem, self))
                    else:
                        self.children.append(ProcessingInstructionNode(elem, self))
                    if elem.tail is not None:
                        self.children.append(TextNode(elem.tail, self))
        yield from self.children

    def iter_descendants(self, with_self: bool=True) -> Iterator[ChildNodeType]:
        if with_self:
            yield self
        for child in self:
            if isinstance(child, ElementNode):
                yield from child.iter_descendants()
            else:
                yield child