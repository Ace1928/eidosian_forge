from typing import Any, Dict, Optional, Union, Iterable, List, Type, TYPE_CHECKING
from lazyops.utils.lazy import get_keydb_session
from .base import BaseStatefulBackend, SchemaType, logger
def _fetch_set_keys(self, decode: Optional[bool]=True) -> List[str]:
    """
        This is a utility func for non-hset
        """
    keys: List[Union[str, bytes]] = self.cache.client.keys(f'{self.base_key}{self.keyjoin}*')
    if decode:
        return [key.decode() if isinstance(key, bytes) else key for key in keys]
    return keys