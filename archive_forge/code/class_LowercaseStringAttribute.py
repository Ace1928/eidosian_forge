import dataclasses
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
class LowercaseStringAttribute(GitlabAttribute):

    def get_for_api(self, *, key: str) -> Tuple[str, str]:
        return (key, str(self._value).lower())