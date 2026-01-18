import os
import pathlib
import torch
from torch.jit._serialization import validate_map_location
def _load_for_lite_interpreter(f, map_location=None):
    """
    Load a :class:`LiteScriptModule` saved with :func:`torch.jit._save_for_lite_interpreter`.

    Args:
        f: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location: a string or torch.device used to dynamically remap
            storages to an alternative set of devices.

    Returns:
        A :class:`LiteScriptModule` object.

    Example:

    .. testcode::

        import torch
        import io

        # Load LiteScriptModule from saved file path
        torch.jit._load_for_lite_interpreter('lite_script_module.pt')

        # Load LiteScriptModule from io.BytesIO object
        with open('lite_script_module.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Load all tensors to the original device
        torch.jit.mobile._load_for_lite_interpreter(buffer)
    """
    if isinstance(f, str):
        if not os.path.exists(f):
            raise ValueError(f'The provided filename {f} does not exist')
        if os.path.isdir(f):
            raise ValueError(f'The provided filename {f} is a directory')
    map_location = validate_map_location(map_location)
    if isinstance(f, (str, pathlib.Path)):
        cpp_module = torch._C._load_for_lite_interpreter(f, map_location)
    else:
        cpp_module = torch._C._load_for_lite_interpreter_from_buffer(f.read(), map_location)
    return LiteScriptModule(cpp_module)