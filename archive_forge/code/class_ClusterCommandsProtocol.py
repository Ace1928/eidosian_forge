from datetime import datetime, timedelta
from typing import (
from redis.compat import Protocol
class ClusterCommandsProtocol(CommandsProtocol, Protocol):
    encoder: 'Encoder'

    def execute_command(self, *args, **options) -> Union[Any, Awaitable]:
        ...