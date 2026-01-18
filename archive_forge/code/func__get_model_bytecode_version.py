import os
import pathlib
import torch
from torch.jit._serialization import validate_map_location
def _get_model_bytecode_version(f_input) -> int:
    """Take a file-like object to return an integer.

    Args:
        f_input: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name

    Returns:
        version: An integer. If the integer is -1, the version is invalid. A warning
            will show in the log.

    Example:
    .. testcode::

        from torch.jit.mobile import _get_model_bytecode_version

        # Get bytecode version from a saved file path
        version = _get_model_bytecode_version("path/to/model.ptl")

    """
    if isinstance(f_input, str):
        if not os.path.exists(f_input):
            raise ValueError(f'The provided filename {f_input} does not exist')
        if os.path.isdir(f_input):
            raise ValueError(f'The provided filename {f_input} is a directory')
    if isinstance(f_input, (str, pathlib.Path)):
        return torch._C._get_model_bytecode_version(str(f_input))
    else:
        return torch._C._get_model_bytecode_version_from_buffer(f_input.read())