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
class JsonResolver(Protocol):
    """Protocol for json resolver functions passed to read_json."""

    def __call__(self, cirq_type: str) -> Optional[ObjectFactory]:
        ...