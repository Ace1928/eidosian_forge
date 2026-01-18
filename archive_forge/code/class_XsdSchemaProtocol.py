from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class XsdSchemaProtocol(XsdValidatorProtocol, ElementProtocol, Protocol):

    def __iter__(self) -> Iterator[XsdXPathNodeType]:
        ...

    def find(self, path: str, namespaces: Optional[Dict[str, str]]=...) -> Optional[XsdXPathNodeType]:
        ...

    def iter(self, tag: Optional[str]=...) -> Iterator[XsdXPathNodeType]:
        ...

    @property
    def tag(self) -> str:
        ...

    @property
    def attrib(self) -> MutableMapping[Optional[str], 'XsdAttributeProtocol']:
        ...