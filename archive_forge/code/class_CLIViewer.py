from io import BytesIO
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from ase.io.formats import ioformats
from ase.io import write
class CLIViewer:

    def __init__(self, name, fmt, argv):
        self.name = name
        self.fmt = fmt
        self.argv = argv

    @property
    def ioformat(self):
        return ioformats[self.fmt]

    @contextmanager
    def mktemp(self, atoms, data=None):
        ioformat = self.ioformat
        suffix = '.' + ioformat.extensions[0]
        if ioformat.isbinary:
            mode = 'wb'
        else:
            mode = 'w'
        with tempfile.TemporaryDirectory(prefix='ase-view-') as dirname:
            path = Path(dirname) / f'atoms{suffix}'
            with path.open(mode) as fd:
                if data is None:
                    write(fd, atoms, format=self.fmt)
                else:
                    write(fd, atoms, format=self.fmt, data=data)
            yield path

    def view_blocking(self, atoms, data=None):
        with self.mktemp(atoms, data) as path:
            subprocess.check_call(self.argv + [str(path)])

    def view(self, atoms, data=None, repeat=None):
        """Spawn a new process in which to open the viewer."""
        if repeat is not None:
            atoms = atoms.repeat(repeat)
        proc = subprocess.Popen([sys.executable, '-m', 'ase.visualize.external'], stdin=subprocess.PIPE)
        pickle.dump((self, atoms, data), proc.stdin)
        proc.stdin.close()
        return proc

    @classmethod
    def viewers(cls):
        return [cls('ase_gui_cli', 'traj', [sys.executable, '-m', 'ase.gui']), cls('avogadro', 'cube', ['avogadro']), cls('gopenmol', 'extxyz', ['runGOpenMol']), cls('rasmol', 'proteindatabank', ['rasmol', '-pdb']), cls('vmd', 'cube', ['vmd']), cls('xmakemol', 'extxyz', ['xmakemol', '-f'])]