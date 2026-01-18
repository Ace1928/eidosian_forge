import re
from qiskit.result import postprocess
from qiskit import exceptions
@staticmethod
def _remove_space_underscore(bitstring):
    """Removes all spaces and underscores from bitstring"""
    return int(bitstring.replace(' ', '').replace('_', ''), 2)