import math
from enum import Enum, unique
from typing import Dict, List, Sequence, Tuple, Union
def _replace_operation_or_retain(operation: _EditOperations, _flip_operations: Dict[_EditOperations, _EditOperations]) -> _EditOperations:
    if operation in _flip_operations:
        return _flip_operations.get(operation)
    return operation