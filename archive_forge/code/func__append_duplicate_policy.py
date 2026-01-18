from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_duplicate_policy(params: List[str], command: Optional[str], duplicate_policy: Optional[str]):
    """Append DUPLICATE_POLICY property to params on CREATE
        and ON_DUPLICATE on ADD.
        """
    if duplicate_policy is not None:
        if command == 'TS.ADD':
            params.extend(['ON_DUPLICATE', duplicate_policy])
        else:
            params.extend(['DUPLICATE_POLICY', duplicate_policy])