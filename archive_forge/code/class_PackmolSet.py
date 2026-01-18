from __future__ import annotations
import os
import subprocess
from pathlib import Path
from shutil import which
import numpy as np
from pymatgen.core import Molecule
from pymatgen.io.core import InputGenerator, InputSet
class PackmolSet(InputSet):
    """
    InputSet for the Packmol software. This class defines several attributes related to.
    """

    def run(self, path: str | Path, timeout=30):
        """Run packmol and write out the packed structure.

        Args:
            path: The path in which packmol input files are located.
            timeout: Timeout in seconds.

        Raises:
            ValueError if packmol does not succeed in packing the box.
            TimeoutExpiredError if packmold does not finish within the timeout.
        """
        wd = os.getcwd()
        if not which('packmol'):
            raise RuntimeError("Running a PackmolSet requires the executable 'packmol' to be in the path. Please download packmol from https://github.com/leandromartinez98/packmol and follow the instructions in the README to compile. Don't forget to add the packmol binary to your path")
        try:
            os.chdir(path)
            p = subprocess.run(f'packmol < {self.inputfile!r}', check=True, shell=True, timeout=timeout, capture_output=True)
            if 'ERROR' in p.stdout.decode():
                if 'Could not open file.' in p.stdout.decode():
                    raise ValueError('Your packmol might be too old to handle paths with spaces.Please try again with a newer version or use paths without spaces.')
                msg = p.stdout.decode().split('ERROR')[-1]
                raise ValueError(f'Packmol failed with return code 0 and stdout: {msg}')
        except subprocess.CalledProcessError as exc:
            raise ValueError(f'Packmol failed with error code {exc.returncode} and stderr: {exc.stderr}') from exc
        else:
            with open(Path(path, self.stdoutfile), mode='w') as out:
                out.write(p.stdout.decode())
        finally:
            os.chdir(wd)

    @classmethod
    def from_directory(cls, directory: str | Path) -> None:
        """
        Construct an InputSet from a directory of one or more files.

        Args:
            directory (str | Path): Directory to read input files from.
        """
        raise NotImplementedError(f'from_directory has not been implemented in {cls.__name__}')