import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, List, Optional, Set, Union
from weakref import WeakSet
import torch._guards
import torch.distributed as dist
def _init_logs(log_file_name=None):
    _reset_logs()
    _update_log_state_from_env()
    for log_qname in log_registry.get_log_qnames():
        if log_qname == 'torch':
            continue
        log = logging.getLogger(log_qname)
        log.setLevel(logging.NOTSET)
    for log_qname, level in log_state.get_log_level_pairs():
        log = logging.getLogger(log_qname)
        log.setLevel(level)
    for log_qname in log_registry.get_log_qnames():
        log = logging.getLogger(log_qname)
        _setup_handlers(logging.StreamHandler, log)
        if log_file_name is not None:
            _setup_handlers(lambda: logging.FileHandler(log_file_name), log)
    for artifact_log_qname in log_registry.get_artifact_log_qnames():
        log = logging.getLogger(artifact_log_qname)
        configure_artifact_log(log)