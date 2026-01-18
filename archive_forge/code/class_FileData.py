from __future__ import annotations
from typing import Any, TypedDict
from typing_extensions import NotRequired
class FileData(TypedDict):
    name: str | None
    data: str | None
    size: NotRequired[int | None]
    is_file: NotRequired[bool]
    orig_name: NotRequired[str]
    mime_type: NotRequired[str]
    is_stream: NotRequired[bool]