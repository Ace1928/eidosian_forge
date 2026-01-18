from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _generate_core_wf_block(self):
    """Create a default onetep core wavefunctions block, using 'NONE'
        unless the user has set overrides for specific species by setting
        specific entries in species_core_wf. If all are NONE, no block
        will be printed"""
    any_core_wfs = False
    for sp in self.species:
        try:
            core_wf_string = self.parameters['species_core_wf'][sp[0]]
            any_core_wfs = True
        except KeyError:
            core_wf_string = 'NONE'
        self.core_wfs.append((sp[0], core_wf_string))
    if not any_core_wfs:
        self.core_wfs = []