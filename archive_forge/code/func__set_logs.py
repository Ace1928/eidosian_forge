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
def _set_logs(**kwargs):
    for alias, val in itertools.chain(kwargs.items(), modules.items()):
        if val is None:
            continue
        if log_registry.is_artifact(alias):
            if not isinstance(val, bool):
                raise ValueError(f'Expected bool to enable artifact {alias}, received {val}')
            if val:
                log_state.enable_artifact(alias)
        elif log_registry.is_log(alias) or alias in log_registry.child_log_qnames:
            if val not in logging._levelToName:
                raise ValueError(f'Unrecognized log level for log {alias}: {val}, valid level values are: {','.join([str(k) for k in logging._levelToName.keys()])}')
            log_state.enable_log(log_registry.log_alias_to_log_qnames.get(alias, alias), val)
        else:
            raise ValueError(f'Unrecognized log or artifact name passed to set_logs: {alias}')
    _init_logs()