import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
class OctNamespace:

    def __init__(self):
        self.names = {}
        self.consts = {'pi': np.pi, 'angstrom': 1.0 / Bohr, 'ev': 1.0 / Hartree, 'yes': True, 'no': False, 't': True, 'f': False, 'i': 1j, 'true': True, 'false': False}

    def evaluate(self, value):
        value = value.strip()
        for char in ('"', "'"):
            if value.startswith(char):
                assert value.endswith(char)
                return value
        value = value.lower()
        if value in self.consts:
            return self.consts[value]
        if value in self.names:
            return self.names[value]
        try:
            v = int(value)
        except ValueError:
            pass
        else:
            if v == float(v):
                return v
        try:
            return float(value)
        except ValueError:
            pass
        if '*' in value or ('/' in value and (not any((char in value for char in '()+')))):
            floatvalue = 1.0
            op = '*'
            for token in re.split('([\\*/])', value):
                if token in '*/':
                    op = token
                    continue
                v = self.evaluate(token)
                try:
                    v = float(v)
                except TypeError:
                    try:
                        v = complex(v)
                    except ValueError:
                        break
                except ValueError:
                    break
                else:
                    if op == '*':
                        floatvalue *= v
                    else:
                        assert op == '/', op
                        floatvalue /= v
            else:
                return floatvalue
        return value

    def add(self, name, value):
        value = self.evaluate(value)
        self.names[name.lower().strip()] = value