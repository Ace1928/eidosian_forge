from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class GlobalMapsProtocol(Protocol):

    @property
    def types(self) -> Mapping[str, XsdGlobalValue[XsdTypeProtocol]]:
        ...

    @property
    def attributes(self) -> Mapping[str, XsdGlobalValue[XsdAttributeProtocol]]:
        ...

    @property
    def elements(self) -> Mapping[str, XsdGlobalValue[XsdElementProtocol]]:
        ...

    @property
    def substitution_groups(self) -> Mapping[str, Set[Any]]:
        ...