from typing import Any, Dict, Optional, Sequence
import dataclasses
from typing_extensions import Protocol
import cirq
Class for specifying device parameters.

    For instance, varying the length of pulses, timing, etc.
    This class is intended to be attached to a cirq.Points
    or cirq.Linspace sweep object as a metadata attribute.

    Args:
       path: path of the key to modify, with each sub-folder as a string
           entry in a list.
       idx: If this key is an array, which index to modify.
       value: value of the parameter to be set, if any.
       units: string value of the unit type of the value, if any.
          For instance, "GHz", "MHz", "ns", etc.
    