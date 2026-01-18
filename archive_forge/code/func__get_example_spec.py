import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def _get_example_spec(name='example-program'):
    return KeyValueExecutableSpec.from_dict(dict(name=name), executable_family='cirq_google.algo_benchmarks.example')