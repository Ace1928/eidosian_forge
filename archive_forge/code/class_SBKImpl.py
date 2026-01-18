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
class SBKImpl(cirq.SerializableByKey):
    """A test implementation of SerializableByKey."""

    def __init__(self, name: str, data_list: Optional[List]=None, data_tuple: Optional[Tuple]=None, data_dict: Optional[Dict]=None):
        self.name = name
        self.data_list = data_list or []
        self.data_tuple = data_tuple or ()
        self.data_dict = data_dict or {}

    def __eq__(self, other):
        if not isinstance(other, SBKImpl):
            return False
        return self.name == other.name and self.data_list == other.data_list and (self.data_tuple == other.data_tuple) and (self.data_dict == other.data_dict)

    def _json_dict_(self):
        return {'name': self.name, 'data_list': self.data_list, 'data_tuple': self.data_tuple, 'data_dict': self.data_dict}

    @classmethod
    def _from_json_dict_(cls, name, data_list, data_tuple, data_dict, **kwargs):
        return cls(name, data_list, tuple(data_tuple), data_dict)