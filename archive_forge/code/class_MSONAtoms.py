from __future__ import annotations
import warnings
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.core.structure import Molecule, Structure
class MSONAtoms(Atoms, MSONable):
    """A custom subclass of ASE Atoms that is MSONable, including `.as_dict()` and `.from_dict()` methods."""

    def as_dict(atoms: Atoms) -> dict[str, Any]:
        atoms_no_info = atoms.copy()
        atoms_no_info.info = {}
        return {'@module': 'pymatgen.io.ase', '@class': 'MSONAtoms', 'atoms_json': encode(atoms_no_info), 'atoms_info': jsanitize(atoms.info, strict=True)}

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        mson_atoms = cls(decode(dct['atoms_json']))
        atoms_info = MontyDecoder().process_decoded(dct['atoms_info'])
        mson_atoms.info = atoms_info
        return mson_atoms