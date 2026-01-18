from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
def _parse_sefv(self) -> list[dict]:
    """
        Parse the MOVEMENT file, return information ionic step structure containing
        structures, energies, forces on atoms and virial tensor.

        Returns:
            list[dict]: Structure containing structures, energies, forces on atoms
                and virial tensor. The corresponding keys are 'atom_config', 'e_tot',
                'atom_forces' and 'virial'.
        """
    ionic_steps: list[dict] = []
    with zopen(self.filename, 'rt') as mvt:
        tmp_step: dict = {}
        for ii in range(self.n_ionic_steps):
            tmp_chunk: str = ''
            for _ in range(self.chunk_sizes[ii]):
                tmp_chunk += mvt.readline()
            tmp_step.update({'atom_config': AtomConfig.from_str(tmp_chunk)})
            tmp_step.update({'e_tot': ACstrExtractor(tmp_chunk).get_e_tot()[0]})
            tmp_step.update({'atom_forces': ACstrExtractor(tmp_chunk).get_atom_forces().reshape(-1, 3)})
            e_atoms: np.ndarray | None = ACstrExtractor(tmp_chunk).get_atom_forces()
            if e_atoms is not None:
                tmp_step.update({'atom_energies': ACstrExtractor(tmp_chunk).get_atom_energies()})
            else:
                print(f'Ionic step #{ii} : Energy deposition is turn down.')
            virial: np.ndarray | None = ACstrExtractor(tmp_chunk).get_virial()
            if virial is not None:
                tmp_step.update({'virial': virial.reshape(3, 3)})
            else:
                print(f'Ionic step #{ii} : No virial information.')
            ionic_steps.append(tmp_step)
    return ionic_steps