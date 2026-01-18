import contextlib
import dataclasses
import datetime
import importlib
import io
import json
import os
import pathlib
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Type
from unittest import mock
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import sympy
import cirq
from cirq._compat import proper_eq
from cirq.protocols import json_serialization
from cirq.testing.json import ModuleJsonTestSpec, spec_for, assert_json_roundtrip_works
def _eval_repr_data_file(path: pathlib.Path, deprecation_deadline: Optional[str]):
    content = path.read_text()
    ctx_managers: List[contextlib.AbstractContextManager] = [contextlib.suppress()]
    if deprecation_deadline:
        ctx_managers = [cirq.testing.assert_deprecated(deadline=deprecation_deadline, count=None)]
    for deprecation in TESTED_MODULES.values():
        if deprecation is not None and deprecation.old_name in content:
            ctx_managers.append(deprecation.deprecation_assertion)
    imports = {'cirq': cirq, 'pd': pd, 'sympy': sympy, 'np': np, 'datetime': datetime, 'nx': nx}
    for m in TESTED_MODULES.keys():
        try:
            imports[m] = importlib.import_module(m)
        except ImportError:
            pass
    with contextlib.ExitStack() as stack:
        for ctx_manager in ctx_managers:
            stack.enter_context(ctx_manager)
        obj = eval(content, imports, {})
        return obj