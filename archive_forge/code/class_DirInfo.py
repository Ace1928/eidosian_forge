import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
class DirInfo:
    name = 'dir_info'

    def __init__(self, editable: bool=False) -> None:
        self.editable = editable

    @classmethod
    def _from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional['DirInfo']:
        if d is None:
            return None
        return cls(editable=_get_required(d, bool, 'editable', default=False))

    def _to_dict(self) -> Dict[str, Any]:
        return _filter_none(editable=self.editable or None)