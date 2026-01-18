import argparse
import copy
import json
import os
import re
import sys
import yaml
import wandb
from wandb import trigger
from wandb.util import add_import_hook, get_optional_module
def _monkey_absl():
    from absl import app as absl_app

    def _absl_callback():
        absl_flags = sys.modules.get('absl.flags')
        if not absl_flags:
            return
        _flags = getattr(absl_flags, 'FLAGS', None)
        if not _flags:
            return
        _flags_as_dict = getattr(_flags, 'flag_values_dict', None)
        if not _flags_as_dict:
            return
        _flags_module = getattr(_flags, 'find_module_defining_flag', None)
        if not _flags_module:
            return
        flags_dict = {}
        for f, v in _flags_as_dict().items():
            m = _flags_module(f)
            if not m or m.startswith('absl.'):
                continue
            flags_dict[f] = v
        global _args_absl
        _args_absl = flags_dict
    call_after_init = getattr(absl_app, 'call_after_init', None)
    if not call_after_init:
        return
    call_after_init(_absl_callback)