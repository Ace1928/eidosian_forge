import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
class PySocketIOClient:

    def __init__(self, calculator_factory):
        self._calculator_factory = calculator_factory

    def __call__(self, atoms, properties=None, port=None, unixsocket=None):
        import sys
        import pickle
        transferbytes = pickle.dumps([dict(unixsocket=unixsocket, port=port), atoms.copy(), self._calculator_factory])
        proc = Popen([sys.executable, '-m', 'ase.calculators.socketio'], stdin=PIPE)
        proc.stdin.write(transferbytes)
        proc.stdin.close()
        return proc

    @staticmethod
    def main():
        import sys
        import pickle
        socketinfo, atoms, get_calculator = pickle.load(sys.stdin.buffer)
        atoms.calc = get_calculator()
        client = SocketClient(host='localhost', unixsocket=socketinfo.get('unixsocket'), port=socketinfo.get('port'))
        client.run(atoms, use_stress=True)