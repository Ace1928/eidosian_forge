import pytest
import cirq
def _with_key_path_(self, path):
    return MultiKeyGate([str(key._with_key_path_(path)) for key in self._keys])