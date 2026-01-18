import sys
import types
from typing import Tuple, Union
def _call_with_lazy_import(fn_name: str, protobuf_path: str) -> Union[types.ModuleType, Tuple[types.ModuleType, types.ModuleType]]:
    """Calls one of the three functions, lazily importing grpc_tools.

    Args:
      fn_name: The name of the function to import from grpc_tools.protoc.
      protobuf_path: The path to import.

    Returns:
      The appropriate module object.
    """
    if sys.version_info < _MINIMUM_VERSION:
        raise NotImplementedError(_VERSION_ERROR_TEMPLATE.format(fn_name))
    else:
        if not _is_grpc_tools_importable():
            raise NotImplementedError(_UNINSTALLED_TEMPLATE.format(fn_name))
        import grpc_tools.protoc
        if _has_runtime_proto_symbols(grpc_tools.protoc):
            fn = getattr(grpc_tools.protoc, '_' + fn_name)
            return fn(protobuf_path)
        else:
            raise NotImplementedError(_UNINSTALLED_TEMPLATE.format(fn_name))