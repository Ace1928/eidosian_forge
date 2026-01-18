from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class DocumentProtocol(Hashable, Protocol):

    def getroot(self) -> Optional[ElementProtocol]:
        ...

    def parse(self, source: Any, *args: Any, **kwargs: Any) -> ElementProtocol:
        ...

    def iter(self, tag: Optional[str]=...) -> Iterator[ElementProtocol]:
        ...