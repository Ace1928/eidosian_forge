import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
from torch.distributed.elastic.utils.logging import get_logger
@staticmethod
def _set_master_addr_port(store: Store, master_addr: Optional[str], master_port: Optional[int], local_addr: Optional[str]):
    if master_port is None:
        sock = _get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
    if master_addr is None:
        if local_addr:
            master_addr = local_addr
        else:
            master_addr = _get_fq_hostname()
    store.set('MASTER_ADDR', master_addr.encode(encoding='UTF-8'))
    store.set('MASTER_PORT', str(master_port).encode(encoding='UTF-8'))