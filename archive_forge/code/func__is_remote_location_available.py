import importlib.machinery
import os
from torch.hub import _get_torch_home
def _is_remote_location_available() -> bool:
    return False