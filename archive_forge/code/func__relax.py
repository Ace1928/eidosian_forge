from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
def _relax(self, calculator: str | Calculator, relax_cell: bool=True, optimizer: str | Optimizer='FIRE', steps: int=500, fmax: float=0.1, stress_weight: float=0.01, opt_kwargs: dict | None=None, return_trajectory: bool=False, verbose: bool=False) -> Structure | Molecule | tuple[Structure | Molecule, TrajectoryObserver | Trajectory]:
    """Performs a structure relaxation using an ASE calculator.

        Args:
            calculator (str | ase.Calculator): An ASE Calculator or a string from the following options: "M3GNet",
                "gfn2-xtb".
            relax_cell (bool): whether to relax the lattice cell. Defaults to True.
            optimizer (str): name of the ASE optimizer class to use
            steps (int): max number of steps for relaxation. Defaults to 500.
            fmax (float): total force tolerance for relaxation convergence.
                Here fmax is a sum of force and stress forces. Defaults to 0.1.
            stress_weight (float): the stress weight for relaxation with M3GNet.
                Defaults to 0.01.
            opt_kwargs (dict): kwargs for the ASE optimizer class.
            return_trajectory (bool): Whether to return the trajectory of relaxation.
                Defaults to False.
            verbose (bool): whether to print stdout. Defaults to False.

        Returns:
            Structure | Molecule: Relaxed structure or molecule
        """
    from ase import optimize
    from ase.constraints import ExpCellFilter
    from ase.io import read
    from ase.optimize.optimize import Optimizer
    from pymatgen.io.ase import AseAtomsAdaptor
    opt_kwargs = opt_kwargs or {}
    is_molecule = isinstance(self, Molecule)
    run_uip = isinstance(calculator, str) and calculator.lower() in ('m3gnet', 'chgnet')
    calc_params = {} if is_molecule else dict(stress_weight=stress_weight)
    calculator = self._prep_calculator(calculator, **calc_params)

    def is_ase_optimizer(key):
        return isclass((obj := getattr(optimize, key))) and issubclass(obj, Optimizer)
    valid_keys = [key for key in dir(optimize) if is_ase_optimizer(key)]
    if isinstance(optimizer, str):
        if optimizer not in valid_keys:
            raise ValueError(f'Unknown optimizer={optimizer!r}, must be one of {valid_keys}')
        opt_class = getattr(optimize, optimizer)
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(self)
    if return_trajectory:
        if run_uip:
            from matgl.ext.ase import TrajectoryObserver
            traj_observer = TrajectoryObserver(atoms)
        else:
            opt_kwargs.setdefault('trajectory', 'opt.traj')
    atoms.calc = calculator
    stream = sys.stdout if verbose else io.StringIO()
    with contextlib.redirect_stdout(stream):
        if relax_cell:
            if is_molecule:
                raise ValueError("Can't relax cell for a Molecule.")
            ecf = ExpCellFilter(atoms)
            dyn = opt_class(ecf, **opt_kwargs)
        else:
            dyn = opt_class(atoms, **opt_kwargs)
        dyn.run(fmax=fmax, steps=steps)
    system: Structure | Molecule = adaptor.get_molecule(atoms) if is_molecule else adaptor.get_structure(atoms)
    system.calc = atoms.calc
    system.dynamics = dyn.todict()
    if return_trajectory:
        if run_uip:
            traj_observer()
        else:
            traj_file = opt_kwargs['trajectory']
            traj_observer = read(traj_file, index=':')
        return (system, traj_observer)
    return system