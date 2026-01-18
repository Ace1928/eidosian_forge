import pytest
import cirq
class SomeRandomClass:

    def __init__(self, some_str):
        self.some_str = some_str

    def __str__(self):
        return self.some_str