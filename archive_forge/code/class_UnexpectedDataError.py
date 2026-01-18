from typing import Any, Type, Optional, Set, Dict
class UnexpectedDataError(DaciteError):

    def __init__(self, keys: Set[str]) -> None:
        super().__init__()
        self.keys = keys

    def __str__(self) -> str:
        formatted_keys = ', '.join((f'"{key}"' for key in self.keys))
        return f'can not match {formatted_keys} to any data class field'