import collections
import datetime
import logging
import os
import threading
from typing import (
import grpc
from grpc.experimental import experimental_api
def _create_channel(target: str, options: Sequence[Tuple[str, str]], channel_credentials: Optional[grpc.ChannelCredentials], compression: Optional[grpc.Compression]) -> grpc.Channel:
    _LOGGER.debug(f"Creating secure channel with credentials '{channel_credentials}', " + f"options '{options}' and compression '{compression}'")
    return grpc.secure_channel(target, credentials=channel_credentials, options=options, compression=compression)