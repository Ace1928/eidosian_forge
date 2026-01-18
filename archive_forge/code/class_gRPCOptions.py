import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@PublicAPI(stability='alpha')
class gRPCOptions(BaseModel):
    """gRPC options for the proxies. Supported fields:

    Args:
        port (int):
            Port for gRPC server if started. Default to 9000. Cannot be
            updated once Serve has started running. Serve must be shut down and
            restarted with the new port instead.
        grpc_servicer_functions (List[str]):
            List of import paths for gRPC `add_servicer_to_server` functions to add to
            Serve's gRPC proxy. Default to empty list, which means no gRPC methods will
            be added and no gRPC server will be started. The servicer functions need to
            be importable from the context of where Serve is running.
    """
    port: int = DEFAULT_GRPC_PORT
    grpc_servicer_functions: List[str] = []

    @property
    def grpc_servicer_func_callable(self) -> List[Callable]:
        """Return a list of callable functions from the grpc_servicer_functions.

        If the function is not callable or not found, it will be ignored and a warning
        will be logged.
        """
        callables = []
        for func in self.grpc_servicer_functions:
            try:
                imported_func = import_attr(func)
                if callable(imported_func):
                    callables.append(imported_func)
                else:
                    message = f'{func} is not a callable function! Please make sure the function is imported correctly.'
                    raise ValueError(message)
            except ModuleNotFoundError as e:
                message = f"{func} can't be imported! Please make sure there are no typo in those functions. Or you might want to rebuild service definitions if .proto file is changed."
                raise ModuleNotFoundError(message) from e
        return callables