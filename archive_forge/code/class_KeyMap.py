import string
from typing import TypeVar, Generic, Tuple, Dict
class KeyMap(Generic[T]):

    def __init__(self, default: T) -> None:
        self.map: Dict[str, T] = {}
        self.default = default

    def __getitem__(self, key: str) -> T:
        if not key:
            return self.default
        elif key in self.map:
            return self.map[key]
        else:
            raise KeyError(f'Configured keymap ({key}) does not exist in bpython.keys')

    def __delitem__(self, key: str):
        del self.map[key]

    def __setitem__(self, key: str, value: T):
        self.map[key] = value