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
class SerializableByKey(SupportsJSON):
    """Protocol for objects that can be serialized to a key + context.

    In serialization, objects that inherit from this type will only be fully
    defined once (the "context"). Thereafter, a unique integer key will be used
    to identify that object.
    """