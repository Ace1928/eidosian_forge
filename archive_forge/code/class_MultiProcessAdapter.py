import collections
import logging
import sys
from typing import Any, Dict, List, MutableMapping, Set, Tuple
import torch
import torch.distributed as dist
class MultiProcessAdapter(logging.LoggerAdapter):
    """
    Creates an adapter to make logging for multiple processes cleaner
    """

    def process(self, msg: str, kwargs: Any) -> Tuple[str, MutableMapping[str, Any]]:
        process_num = kwargs.pop('process_num', self.extra['process_num'])
        return (f'process: {process_num} {msg}', kwargs)