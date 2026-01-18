from typing import overload, Any, Dict, Iterator, Iterable, Optional, Sequence, ItemsView, \
class LxmlDocumentProtocol(Hashable, Protocol):

    def getroot(self) -> Optional[LxmlElementProtocol]:
        ...

    def parse(self, source: Any, *args: Any, **kwargs: Any) -> LxmlElementProtocol:
        ...

    def iter(self, tag: Optional[str]=...) -> Iterator[LxmlElementProtocol]:
        ...