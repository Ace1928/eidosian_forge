import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
class DecomposeGenerated:

    def _decompose_(self):
        yield cirq.X(cirq.LineQubit(0))
        yield cirq.Y(cirq.LineQubit(1))