import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
@classmethod
def _create_parameters_dict(cls, parameters: List[Tuple[cirq.Qid, cirq.Qid, PhasedFSimCharacterization]]) -> Dict[Tuple[cirq.Qid, cirq.Qid], PhasedFSimCharacterization]:
    """Utility function to create parameters from JSON.

        Can be used from child classes to instantiate classes in a _from_json_dict_
        method."""
    return {(q_a, q_b): params for q_a, q_b, params in parameters}