from io import BytesIO
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from ase.io.formats import ioformats
from ase.io import write
class PyViewer:

    def __init__(self, name, supports_repeat=False):
        self.name = name
        self.supports_repeat = supports_repeat

    def view(self, atoms, data=None, repeat=None):
        func = getattr(self, self.name)
        if self.supports_repeat:
            return func(atoms, repeat)
        else:
            if repeat is not None:
                atoms = atoms.repeat(repeat)
            return func(atoms)

    def sage(self, atoms):
        from ase.visualize.sage import view_sage_jmol
        return view_sage_jmol(atoms)

    def ngl(self, atoms):
        from ase.visualize.ngl import view_ngl
        return view_ngl(atoms)

    def x3d(self, atoms):
        from ase.visualize.x3d import view_x3d
        return view_x3d(atoms)

    def ase(self, atoms, repeat):
        return _pipe_to_ase_gui(atoms, repeat)

    @classmethod
    def viewers(cls):
        return [cls('ase', supports_repeat=True), cls('ngl'), cls('sage'), cls('x3d')]