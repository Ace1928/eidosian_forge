from unittest.mock import patch
import copy
import numpy as np
import sympy
import pytest
import cirq
import cirq_pasqal
class MockGet:

    def __init__(self, json):
        self.counter = 0
        self.json = json

    def raise_for_status(self):
        pass

    @property
    def text(self):
        self.counter += 1
        if self.counter > 1:
            return self.json