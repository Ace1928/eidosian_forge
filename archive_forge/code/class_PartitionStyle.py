import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class PartitionStyle(str, Enum):
    """Supported dataset partition styles.

    Inherits from `str` to simplify plain text serialization/deserialization.

    Examples:
        >>> # Serialize to JSON text.
        >>> json.dumps(PartitionStyle.HIVE)  # doctest: +SKIP
        '"hive"'

        >>> # Deserialize from JSON text.
        >>> PartitionStyle(json.loads('"hive"'))  # doctest: +SKIP
        <PartitionStyle.HIVE: 'hive'>
    """
    HIVE = 'hive'
    DIRECTORY = 'dir'