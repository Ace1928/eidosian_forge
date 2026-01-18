import re
from typing import List, Optional, Tuple
from pip._vendor.packaging.tags import (
def _expand_allowed_platforms(platforms: Optional[List[str]]) -> Optional[List[str]]:
    if not platforms:
        return None
    seen = set()
    result = []
    for p in platforms:
        if p in seen:
            continue
        additions = [c for c in _get_custom_platforms(p) if c not in seen]
        seen.update(additions)
        result.extend(additions)
    return result