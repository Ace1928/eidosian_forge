import sys
import types
from typing import Tuple, Union
def _is_grpc_tools_importable() -> bool:
    try:
        import grpc_tools
        return True
    except ImportError as e:
        if 'grpc_tools' not in e.args[0]:
            raise
        return False