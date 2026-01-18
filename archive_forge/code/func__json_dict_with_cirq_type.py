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
def _json_dict_with_cirq_type(obj: Any):
    base_dict = obj._json_dict_()
    if 'cirq_type' in base_dict:
        raise ValueError(f"Found 'cirq_type': '{base_dict['cirq_type']}' in user-specified _json_dict_. 'cirq_type' is now automatically generated from the class's name and its _json_namespace_ method as `cirq_type: '[<namespace>.]<class_name>'`.\n\nStarting in v0.15, custom 'cirq_type' values will trigger an error. To fix this, remove 'cirq_type' from the class _json_dict_ method and define _json_namespace_ for the class.\n\nFor backwards compatibility, third-party classes whose old 'cirq_type' value does not match the new value must appear under BOTH values in the resolver for that package. For details on defining custom resolvers, see the DEFAULT_RESOLVER docstring in cirq-core/cirq/protocols/json_serialization.py.")
    return {'cirq_type': json_cirq_type(type(obj)), **base_dict}