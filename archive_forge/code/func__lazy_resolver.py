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
def _lazy_resolver(dict_factory: Callable[[], Dict[str, ObjectFactory]]) -> JsonResolver:
    """A lazy JsonResolver based on a dict_factory.

    It only calls dict_factory when the first key is accessed.

    Args:
        dict_factory: a callable that generates an instance of the
          class resolution map - it is assumed to be cached
    """

    def json_resolver(cirq_type: str) -> Optional[ObjectFactory]:
        return dict_factory().get(cirq_type, None)
    return json_resolver