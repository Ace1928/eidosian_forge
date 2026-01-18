import json
import numpy as np
import ase.units as units
from ase import Atoms
from ase.data import chemical_symbols
class NomadEntry(dict):
    """An entry from the Nomad database.

    The Nomad entry is represented as nested dictionaries and lists.

    ASE converts each dictionary into a NomadEntry object which supports
    different actions.  Some actions are only available when the NomadEntry
    represents a particular section."""

    def __init__(self, dct):
        dict.__init__(self, dct)

    @property
    def hash(self):
        assert self['uri'].startswith('nmd://')
        return self['uri'][6:]

    def toatoms(self):
        """Convert this NomadEntry into an Atoms object.

        This NomadEntry must represent a section_system."""
        return section_system_to_atoms(self)

    def iterimages(self):
        """Yield Atoms object contained within this NomadEntry.

        This NomadEntry must represent or contain a section_run."""
        if 'section_run' in self:
            run_sections = self['section_run']
        else:
            assert self['name'] == 'section_run'
            run_sections = [self]
        for run in run_sections:
            systems = run['section_system']
            for system in systems:
                atoms = section_system_to_atoms(system)
                atoms.info['nomad_run_gIndex'] = run['gIndex']
                atoms.info['nomad_system_gIndex'] = system['gIndex']
                if self.get('name') == 'calculation_context':
                    atoms.info['nomad_calculation_uri'] = self['uri']
                yield atoms