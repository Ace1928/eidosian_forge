import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
def _dispatch_opset_version(target: OpsetVersion, registered_opsets: Collection[OpsetVersion]) -> Optional[OpsetVersion]:
    """Finds the registered opset given a target opset version and the available opsets.

    Args:
        target: The target opset version.
        registered_opsets: The available opsets.

    Returns:
        The registered opset version.
    """
    if not registered_opsets:
        return None
    descending_registered_versions = sorted(registered_opsets, reverse=True)
    if target >= _constants.ONNX_BASE_OPSET:
        for version in descending_registered_versions:
            if version <= target:
                return version
        return None
    for version in reversed(descending_registered_versions):
        if target <= version <= _constants.ONNX_BASE_OPSET:
            return version
    return None