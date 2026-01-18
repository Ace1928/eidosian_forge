from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
def _setup_task(self, geo_subkeys):
    """
        Setup the block 'Geometry' given subkeys and the task.

        Args:
            geo_subkeys (Sized): User-defined subkeys for the block 'Geometry'.

        Notes:
            Most of the run types of ADF are specified in the Geometry
            block except the 'AnalyticFreq'.
        """
    self.geo = AdfKey('Geometry', subkeys=geo_subkeys)
    if self.operation.lower() == 'energy':
        self.geo.add_option('SinglePoint')
        if self.geo.has_subkey('Frequencies'):
            self.geo.remove_subkey('Frequencies')
    elif self.operation.lower() == 'optimize':
        self.geo.add_option('GeometryOptimization')
        if self.geo.has_subkey('Frequencies'):
            self.geo.remove_subkey('Frequencies')
    elif self.operation.lower() == 'numerical_frequencies':
        self.geo.add_subkey(AdfKey('Frequencies'))
    else:
        self.other_directives.append(AdfKey('AnalyticalFreq'))
        if self.geo.has_subkey('Frequencies'):
            self.geo.remove_subkey('Frequencies')