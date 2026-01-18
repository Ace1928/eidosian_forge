import collections
import functools
import itertools
import logging
import os
import random
import types
import typing
import weakref
from typing import Any, Callable, Dict, List, Optional, Set
import torch
import torch._logging
from torch._guards import compile_context, CompileContext, CompileId, tracing
from torch._utils_internal import log_compilation_event, signpost_event
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph_module import _forward_from_src as original_forward_from_src
from torch.utils._traceback import format_traceback_short
from . import config, exc
from .allowed_functions import is_allowed, is_numpy
from .backends.registry import CompilerFn
from .bytecode_analysis import remove_dead_code, remove_pointless_jumps
from .bytecode_transformation import (
from .cache_size import (
from .eval_frame import always_optimize_code_objects, skip_code, TorchPatcher
from .exc import (
from .guards import (
from .hooks import Hooks
from .output_graph import OutputGraph
from .replay_record import ExecutionRecord
from .symbolic_convert import InstructionTranslator, SpeculationLog
from .types import BytecodeHook
from .utils import (
from collections import OrderedDict
from torch.utils.hooks import RemovableHandle
def _patch_config_if_changed():
    """
    Will return {} if the ambient config is the same as the compile-time.
    Else, returns the compile-time config saved on the code object.
    """
    patch: Dict[str, Any] = {}
    eval_frame = torch._dynamo.eval_frame
    eval_frame._maybe_init_guarded_config_cache()
    if eval_frame.config_cache.saved_config_and_hash is None:
        return patch
    saved = eval_frame.config_cache.saved_config_and_hash
    saved_config, saved_config_hash = (saved.config, saved.hash)
    current_config_hash = config.get_hash()
    assert current_config_hash is not None
    if saved_config_hash != current_config_hash:
        patch = saved_config
        if recompiles_log.isEnabledFor(logging.DEBUG):
            recompiles_log.debug('Current config does not match config saved when compiling\nSaved hash: %s, Current hash: %s\nRestoring saved config.', saved_config_hash.hex() if config.verbose else saved_config_hash.hex()[:7], current_config_hash.hex() if config.verbose else current_config_hash.hex()[:7])
            config_dict_ref = config.shallow_copy_dict()
            for key in patch:
                if patch[key] != config_dict_ref[key]:
                    recompiles_log.debug('* %s=%s (prev: %s)', key, patch[key], config_dict_ref[key])
    return patch