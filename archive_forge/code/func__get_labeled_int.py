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
def _get_labeled_int(key: str, s: str):
    ma = re.match(f'{key}_(\\d+)$', s)
    if ma is None:
        raise ValueError(f'Could not parse {key} value for {s}')
    return int(ma.group(1))