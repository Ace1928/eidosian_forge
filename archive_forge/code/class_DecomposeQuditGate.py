import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
class DecomposeQuditGate:

    def _decompose_(self, qids):
        yield cirq.identity_each(*qids)