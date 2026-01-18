import re
from collections import OrderedDict
from typing import Any, Optional
class CaseFoldedOrderedDict(OrderedDict):

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(_casefold(key))

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(_casefold(key), value)

    def __contains__(self, key: object) -> bool:
        return super().__contains__(_casefold(key))

    def get(self, key: str, default: Optional[Any]=None) -> Any:
        return super().get(_casefold(key), default)

    def pop(self, key: str, default: Optional[Any]=None) -> Any:
        return super().pop(_casefold(key), default)