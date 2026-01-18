from subprocess import Popen, PIPE
from ase.calculators.calculator import Calculator
from ase.io import read
from .create_input import GenerateVaspInput
import time
import os
import sys
def _run_vasp(self, atoms):
    if self.process is None:
        stopcar = os.path.join(self.path, 'STOPCAR')
        if os.path.isfile(stopcar):
            os.remove(stopcar)
        self._stdout('Writing VASP input files\n')
        self.initialize(atoms)
        self.write_input(atoms, directory=self.path)
        self._stdout('Starting VASP for initial step...\n')
        if sys.version_info[0] >= 3:
            self.process = Popen(self.command, stdout=PIPE, stdin=PIPE, stderr=PIPE, cwd=self.path, universal_newlines=True)
        else:
            self.process = Popen(self.command, stdout=PIPE, stdin=PIPE, stderr=PIPE, cwd=self.path)
    else:
        self._stdout('Inputting positions...\n')
        for atom in atoms.get_scaled_positions():
            self._stdin(' '.join(map('{:19.16f}'.format, atom)))
    while self.process.poll() is None:
        text = self.process.stdout.readline()
        self._stdout(text)
        if 'POSITIONS: reading from stdin' in text:
            return
    raise RuntimeError('VASP exited unexpectedly with exit code {}'.format(self.subprocess.poll()))