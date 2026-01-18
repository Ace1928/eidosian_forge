from typing import TYPE_CHECKING, Tuple
class CodingStateMachineDict(TypedDict, total=False):
    class_table: Tuple[int, ...]
    class_factor: int
    state_table: Tuple[int, ...]
    char_len_table: Tuple[int, ...]
    name: str
    language: str