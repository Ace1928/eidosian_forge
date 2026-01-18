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
@functools.lru_cache
def _parse_log_settings(settings):
    if settings == '':
        return dict()
    if settings == 'help':
        raise ValueError(help_message(verbose=False))
    elif settings == '+help':
        raise ValueError(help_message(verbose=True))
    if not _validate_settings(settings):
        raise ValueError(_invalid_settings_err_msg(settings))
    settings = re.sub('\\s+', '', settings)
    log_names = settings.split(',')

    def get_name_level_pair(name):
        clean_name = name.replace(INCR_VERBOSITY_CHAR, '')
        clean_name = clean_name.replace(DECR_VERBOSITY_CHAR, '')
        if name[0] == INCR_VERBOSITY_CHAR:
            level = logging.DEBUG
        elif name[0] == DECR_VERBOSITY_CHAR:
            level = logging.ERROR
        else:
            level = logging.INFO
        return (clean_name, level)
    log_state = LogState()
    for name in log_names:
        name, level = get_name_level_pair(name)
        if name == 'all':
            name = 'torch'
        if log_registry.is_log(name):
            assert level is not None
            log_qnames = log_registry.log_alias_to_log_qnames[name]
            log_state.enable_log(log_qnames, level)
        elif log_registry.is_artifact(name):
            log_state.enable_artifact(name)
        elif _is_valid_module(name):
            if not _has_registered_parent(name):
                log_registry.register_log(name, name)
            else:
                log_registry.register_child_log(name)
            log_state.enable_log(name, level)
        else:
            raise ValueError(_invalid_settings_err_msg(settings))
    return log_state