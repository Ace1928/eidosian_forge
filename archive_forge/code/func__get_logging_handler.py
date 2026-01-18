import logging
from typing import List, Tuple
from torch.distributed._shard.sharded_tensor.logging_handlers import (
def _get_logging_handler(destination: str='default') -> Tuple[logging.Handler, str]:
    log_handler = _log_handlers[destination]
    log_handler_name = type(log_handler).__name__
    return (log_handler, log_handler_name)