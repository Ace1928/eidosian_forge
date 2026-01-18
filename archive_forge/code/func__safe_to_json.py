import abc
import dataclasses
import os
from typing import Any, Dict, List, TYPE_CHECKING
import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
def _safe_to_json(obj: Any, *, part_path: str, nominal_path: str, bak_path: str):
    """Safely update a json file.

    1. The new value is written to a "part" file
    2. The previous file atomically replaces the previous backup file, thereby becoming the
       current backup file.
    3. The part file is atomically renamed to the desired filename.
    """
    cirq.to_json_gzip(obj, part_path)
    if os.path.exists(nominal_path):
        os.replace(nominal_path, bak_path)
    os.replace(part_path, nominal_path)