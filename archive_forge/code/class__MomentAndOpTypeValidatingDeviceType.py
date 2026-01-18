import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
class _MomentAndOpTypeValidatingDeviceType(cirq.Device):

    def validate_operation(self, operation):
        if not isinstance(operation, cirq.Operation):
            raise ValueError(f'not isinstance({operation!r}, {cirq.Operation!r})')

    def validate_moment(self, moment):
        if not isinstance(moment, cirq.Moment):
            raise ValueError(f'not isinstance({moment!r}, {cirq.Moment!r})')