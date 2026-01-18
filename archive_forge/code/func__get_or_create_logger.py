import logging
from typing import List, Tuple
from torch.distributed._shard.sharded_tensor.logging_handlers import (
def _get_or_create_logger() -> logging.Logger:
    logging_handler, log_handler_name = _get_logging_handler()
    logger = logging.getLogger(f'sharding-spec-{log_handler_name}')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s')
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger