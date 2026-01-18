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
def _get_testspecs_for_modules() -> List[ModuleJsonTestSpec]:
    modules = []
    for m in TESTED_MODULES.keys():
        try:
            modules.append(spec_for(m))
        except ModuleNotFoundError:
            pass
    return modules