import abc
from typing import Generic, Iterable, Mapping, NoReturn, Optional, Sequence
import grpc
from ._metadata import Metadata
from ._typing import DoneCallbackType
from ._typing import MetadataType
from ._typing import RequestType
from ._typing import ResponseType
Return True if the RPC is done.

        An RPC is done if the RPC is completed, cancelled or aborted.

        This is an EXPERIMENTAL API.

        Returns:
          A bool indicates if the RPC is done.
        