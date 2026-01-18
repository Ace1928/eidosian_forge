from __future__ import annotations
import itertools
import warnings
from collections.abc import Iterator, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Composition, DummySpecies, Element, Lattice, Molecule, Species, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Vasprun, Xdatcar
def _check_frame_props(self, frame_props: list[dict] | None) -> None:
    """Check data shape of site properties."""
    if frame_props is None:
        return
    assert len(frame_props) == len(self), f'Size of the frame properties {len(frame_props)} does not equal to the number of frames {len(self)}.'