import dataclasses
import datetime
import gzip
import json
import numbers
import pathlib
from typing import (
import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
class HasJSONNamespace(Protocol):
    """An object which prepends a namespace to its JSON cirq_type.

    Classes which implement this method have the following cirq_type format:

        f"{obj._json_namespace_()}.{obj.__class__.__name__}

    Classes outside of Cirq or its submodules MUST implement this method to be
    used in type serialization.
    """

    @doc_private
    @classmethod
    def _json_namespace_(cls) -> str:
        pass