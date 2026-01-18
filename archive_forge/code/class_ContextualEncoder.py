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
class ContextualEncoder(cls):
    """An encoder with a context map for concise serialization."""
    seen: Set[str] = set()

    def default(self, o):
        if not isinstance(o, SerializableByKey):
            return super().default(o)
        for candidate in obj.object_dag[:-1]:
            if candidate.obj == o:
                if not candidate.key in ContextualEncoder.seen:
                    ContextualEncoder.seen.add(candidate.key)
                    return _json_dict_with_cirq_type(candidate.obj)
                else:
                    return _json_dict_with_cirq_type(_SerializedKey(candidate.key))
        raise ValueError('Object mutated during serialization.')