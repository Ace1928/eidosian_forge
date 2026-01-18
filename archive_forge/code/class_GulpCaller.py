from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class GulpCaller:
    """Class to run gulp from command line."""

    def __init__(self, cmd='gulp'):
        """Initialize with the executable if not in the standard path.

        Args:
            cmd: Command. Defaults to gulp.
        """

        def is_exe(f) -> bool:
            return os.path.isfile(f) and os.access(f, os.X_OK)
        fpath, _fname = os.path.split(cmd)
        if fpath:
            if is_exe(cmd):
                self._gulp_cmd = cmd
                return
        else:
            for path in os.environ['PATH'].split(os.pathsep):
                path = path.strip('"')
                file = os.path.join(path, cmd)
                if is_exe(file):
                    self._gulp_cmd = file
                    return
        raise GulpError('Executable not found')

    def run(self, gin):
        """Run GULP using the gin as input.

        Args:
            gin: GULP input string

        Returns:
            gout: GULP output string
        """
        with ScratchDir('.'):
            with subprocess.Popen(self._gulp_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as p:
                out, err = p.communicate(bytearray(gin, 'utf-8'))
            out = out.decode('utf-8')
            err = err.decode('utf-8')
            if 'Error' in err or 'error' in err:
                print(gin)
                print('----output_0---------')
                print(out)
                print('----End of output_0------\n\n\n')
                print('----output_1--------')
                print(out)
                print('----End of output_1------')
                raise GulpError(err)
            if 'ERROR' in out:
                raise GulpError(out)
            conv_err_string = 'Conditions for a minimum have not been satisfied'
            if conv_err_string in out:
                raise GulpConvergenceError(out)
            gout = ''
            for line in out.split('\n'):
                gout = gout + line + '\n'
            return gout