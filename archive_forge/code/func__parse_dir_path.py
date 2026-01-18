import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
def _parse_dir_path(self, dir_path: str) -> Dict[str, str]:
    """Directory partition path parser.

        Returns a dictionary mapping directory partition keys to values from a
        partition path of the form "{value1}/{value2}/..." or an empty dictionary for
        unpartitioned files.

        Requires a corresponding ordered list of partition key field names to map the
        correct key to each value.
        """
    dirs = [d for d in dir_path.split('/') if d]
    field_names = self._scheme.field_names
    if dirs and len(dirs) != len(field_names):
        raise ValueError(f'Expected {len(field_names)} partition value(s) but found {len(dirs)}: {dirs}.')
    if not dirs:
        return {}
    return {field: directory for field, directory in zip(field_names, dirs) if field is not None}