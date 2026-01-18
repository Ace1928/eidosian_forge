import sys
import types
from typing import Tuple, Union
Returns a 2-tuple of modules corresponding to protos and services.

    THIS IS AN EXPERIMENTAL API.

    The return value of this function is equivalent to a call to protos and a
    call to services.

    To completely disable the machinery behind this function, set the
    GRPC_PYTHON_DISABLE_DYNAMIC_STUBS environment variable to "true".

    Args:
      protobuf_path: The path to the .proto file on the filesystem. This path
        must be resolveable from an entry on sys.path and so must all of its
        transitive dependencies.

    Returns:
      A 2-tuple of module objects corresponding to (protos(path), services(path)).
    