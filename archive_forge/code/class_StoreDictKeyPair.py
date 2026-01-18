import argparse
import os
import numpy as np
import pytest
from _pytest.runner import pytest_runtest_makereport as orig_pytest_runtest_makereport
import pennylane as qml
class StoreDictKeyPair(argparse.Action):
    """Argparse action for storing key-value pairs as a dictionary.

    For example, calling a CLI program with ``--mydict v1=k1 v2=5``:

    >>> parser.add_argument("--mydict", dest="my_dict", action=StoreDictKeyPair, nargs="+")
    >>> args = parser.parse()
    >>> args.my_dict
    {"v1": "k1", "v2": "5"}

    Note that strings will be converted to ints and floats if possible.

    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split('=')
            my_dict[k] = _convert_to_int_or_float(v)
        setattr(namespace, self.dest, my_dict)