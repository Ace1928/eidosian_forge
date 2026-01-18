import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
class InPlaceUnitary(cirq.testing.SingleQubitGate):

    def _has_unitary_(self):
        return True

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
        args.target_tensor[0], args.target_tensor[1] = (args.target_tensor[1], args.target_tensor[0])
        return args.target_tensor