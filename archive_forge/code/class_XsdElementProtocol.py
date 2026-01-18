from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class XsdElementProtocol(XsdComponentProtocol, ElementProtocol, Protocol):

    def __iter__(self) -> Iterator['XsdElementProtocol']:
        ...

    def find(self, path: str, namespaces: Optional[Dict[str, str]]=...) -> Optional[XsdXPathNodeType]:
        ...

    def iter(self, tag: Optional[str]=...) -> Iterator['XsdElementProtocol']:
        ...

    @property
    def name(self) -> Optional[str]:
        ...

    @property
    def type(self) -> Optional[XsdTypeProtocol]:
        ...

    @property
    def ref(self) -> Optional[Any]:
        ...

    @property
    def attrib(self) -> XsdAttributeGroupProtocol:
        ...