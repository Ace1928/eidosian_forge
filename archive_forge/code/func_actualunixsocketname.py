import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def actualunixsocketname(name):
    return '/tmp/ipi_{}'.format(name)