from __future__ import annotations
import abc
import json
import logging
import os
import warnings
from glob import glob
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.gaussian import GaussianOutput
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar
from pymatgen.io.vasp.outputs import Dynmat, Oszicar, Vasprun
class AbstractDrone(MSONable, abc.ABC):
    """Abstract drone class that defines the various methods that must be
    implemented by drones. Because of the quirky nature of Python"s
    multiprocessing, the intermediate data representations has to be in the
    form of python primitives. So all objects that drones work with must be
    MSONable. All drones must also implement the standard MSONable as_dict() and
    from_dict API.
    """

    @abc.abstractmethod
    def assimilate(self, path):
        """Assimilate data in a directory path into a pymatgen object. Because of
        the quirky nature of Python's multiprocessing, the object must support
        pymatgen's as_dict() for parallel processing.

        Args:
            path: directory path

        Returns:
            An assimilated object
        """
        return

    @abc.abstractmethod
    def get_valid_paths(self, path):
        """Checks if path contains valid data for assimilation, and then returns
        the valid paths. The paths returned can be a list of directory or file
        paths, depending on what kind of data you are assimilating. For
        example, if you are assimilating VASP runs, you are only interested in
        directories containing vasprun.xml files. On the other hand, if you are
        interested converting all POSCARs in a directory tree to CIFs for
        example, you will want the file paths.

        Args:
            path: input path as a tuple generated from os.walk, i.e.,
                (parent, subdirs, files).

        Returns:
            List of valid dir/file paths for assimilation
        """
        return