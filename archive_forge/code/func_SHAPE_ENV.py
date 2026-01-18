from __future__ import annotations
import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import textwrap
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import ReferenceType
import torch
import torch.utils._device
from torch._dynamo.source import (
from torch._guards import (
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef
from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import DefaultsSource, LocalSource, TypeSource
from .types import GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
def SHAPE_ENV(self, guard: Guard):
    assert guard.name == ''
    output_graph = self.check_fn_manager.output_graph
    fs = output_graph.tracked_fakes
    constraint_inputs = [a.constraint_dims for a in fs]

    def get_sources(t_id, dim):
        return [TensorPropertySource(source, TensorProperty.SIZE, dim) for source in output_graph.tracked_fakes_id_to_source[t_id]]
    if output_graph.export_constraints:
        source_pairs: List[Tuple[Source, Source]] = []
        for constraint in output_graph.export_constraints:
            if constraint.t_id in output_graph.tracked_fakes_id_to_source:
                source, *other_sources = get_sources(constraint.t_id, constraint.dim)
                source_pairs.extend(((source, other_source) for other_source in other_sources))
                if constraint.shared is not None:
                    other_sources = get_sources(constraint.shared.t_id, constraint.shared.dim)
                    source_pairs.extend(((source, other_source) for other_source in other_sources))
            else:
                log.warning('Untracked tensor used in export constraints')
        equalities_inputs = EqualityConstraint(source_pairs=source_pairs, warn_only=False)
    else:
        equalities_inputs = None
    guards = output_graph.shape_env.produce_guards([a.fake for a in fs], [a.source for a in fs], constraint_inputs=constraint_inputs, equalities_inputs=equalities_inputs, source_ref=self.source_ref, ignore_static=not self.check_fn_manager.output_graph.export)
    output_graph.shape_env.freeze()
    for shape_guard in guards:
        self._produce_guard_code(guard, [shape_guard], shape_env=True)