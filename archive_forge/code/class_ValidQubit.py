import pytest
import cirq
import cirq_google.api.v2 as v2
class ValidQubit(cirq.Qid):

    def __init__(self, name):
        self._name = name

    @property
    def dimension(self):
        pass

    def _comparison_key(self):
        pass