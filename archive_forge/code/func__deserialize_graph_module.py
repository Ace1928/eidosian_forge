import copy
import itertools
import linecache
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer
from ._compatibility import compatibility
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode
import torch
from torch.nn import *
def _deserialize_graph_module(forward, body: Dict[Any, Any], graph_module_cls=None) -> torch.nn.Module:
    """
    Deserialize a GraphModule given the dictionary of the original module,
    using the code to reconstruct the graph. We delete the actual graph before
    saving the dictionary so that changes to the in-memory graph format do not
    get serialized.
    """
    _CodeOnlyModule.forward = forward
    tracer_cls = body.get('_tracer_cls')
    if tracer_cls is None:
        from ._symbolic_trace import Tracer
        tracer_cls = Tracer
    graphmodule_cls_name = body.get('_graphmodule_cls_name', 'GraphModule')
    cls_tracer: Any = tracer_cls

    class KeepModules(cls_tracer):

        def is_leaf_module(self, _: torch.nn.Module, __: str) -> bool:
            return True
    com = _CodeOnlyModule(body)
    tracer_extras = body.get('_tracer_extras', {})
    graph = KeepModules().trace(com, **tracer_extras)
    graph._tracer_cls = tracer_cls
    if graph_module_cls is None:
        graph_module_cls = GraphModule
    gm = graph_module_cls(com, graph, class_name=graphmodule_cls_name)
    for k, v in body.items():
        if not hasattr(gm, k):
            setattr(gm, k, v)
    return gm