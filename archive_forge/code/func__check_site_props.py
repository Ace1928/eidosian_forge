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
def _check_site_props(self, site_props: SitePropsType | None) -> None:
    """Check data shape of site properties.

        Args:
            site_props (dict | list[dict] | None): Returns immediately if None.

        Raises:
            AssertionError: If the size of the site properties does not match
                the number of sites in the structure.
        """
    if site_props is None:
        return
    if isinstance(site_props, dict):
        site_props = [site_props]
    else:
        assert len(site_props) == len(self), f'Size of the site properties {len(site_props)} does not equal to the number of frames {len(self)}.'
    n_sites = len(self.coords[0])
    for dct in site_props:
        for key, val in dct.items():
            assert len(val) == n_sites, f'Size of site property {key} {len(val)}) does not equal to the number of sites in the structure {n_sites}.'