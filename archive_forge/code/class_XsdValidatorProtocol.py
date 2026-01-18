from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class XsdValidatorProtocol(Hashable, Protocol):

    def is_matching(self, name: Optional[str], default_namespace: Optional[str]=None) -> bool:
        ...

    @property
    def name(self) -> Optional[str]:
        ...

    @property
    def xsd_version(self) -> str:
        ...

    @property
    def maps(self) -> 'GlobalMapsProtocol':
        ...