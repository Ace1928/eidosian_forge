from __future__ import annotations
import itertools
import os
import warnings
import numpy as np
from ruamel.yaml import YAML
from pymatgen.core import SETTINGS
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Element, Molecule, Structure
from pymatgen.io.cp2k.inputs import (
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff, get_unique_site_indices
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
def activate_motion(self, max_drift: float=0.003, rms_drift: float=0.0015, max_force: float=0.00045, rms_force: float=0.0003, max_iter: int=200, optimizer: str='BFGS', trust_radius: float=0.25, line_search: str='2PNT', ensemble: str='NVE', temperature: float=300, timestep: float=0.5, nsteps: int=3, thermostat: str='NOSE', nproc_rep: int=1) -> None:
    """
        Turns on the motion section for GEO_OPT, CELL_OPT, etc. calculations.
        Will turn on the printing subsections and also bind any constraints
        to their respective atoms.
        """
    if not self.check('MOTION'):
        self.insert(Section('MOTION', subsections={}))
    run_type = self['global'].get('run_type', Keyword('run_type', 'energy')).values[0].upper()
    run_type = {'GEOMETRY_OPTIMIZATION': 'GEO_OPT', 'MOLECULAR_DYNAMICS': 'MD'}.get(run_type, run_type)
    self['MOTION'].insert(Section('PRINT', subsections={}))
    self['MOTION']['PRINT'].insert(Section('TRAJECTORY', section_parameters=['ON'], subsections={}))
    self['MOTION']['PRINT'].insert(Section('CELL', subsections={}))
    self['MOTION']['PRINT'].insert(Section('FORCES', subsections={}))
    self['MOTION']['PRINT'].insert(Section('STRESS', subsections={}))
    if run_type in ['GEO_OPT', 'CELL_OPT']:
        opt_params = {'MAX_DR': Keyword('MAX_DR', max_drift), 'MAX_FORCE': Keyword('MAX_FORCE', max_force), 'RMS_DR': Keyword('RMS_DR', rms_drift), 'RMS_FORCE': Keyword('RMS_FORCE', rms_force), 'MAX_ITER': Keyword('MAX_ITER', max_iter), 'OPTIMIZER': Keyword('OPTIMIZER', optimizer)}
        opt = Section(run_type, subsections={}, keywords=opt_params)
        if optimizer.upper() == 'CG':
            ls = Section('LINE_SEARCH', subsections={}, keywords={'TYPE': Keyword('TYPE', line_search)})
            cg = Section('CG', subsections={'LINE_SEARCH': ls}, keywords={})
            opt.insert(cg)
        elif optimizer.upper() == 'BFGS':
            bfgs = Section('BFGS', subsections={}, keywords={'TRUST_RADIUS': Keyword('TRUST_RADIUS', trust_radius)})
            opt.insert(bfgs)
        self['MOTION'].insert(opt)
    elif run_type == 'MD':
        md_keywords = {'ENSEMBLE': Keyword('ENSEMBLE', ensemble), 'TEMPERATURE': Keyword('TEMPERATURE', temperature), 'TIMESTEP': Keyword('TIMESTEP', timestep), 'STEPS': Keyword('STEPS', nsteps)}
        thermostat = Section('THERMOSTAT', keywords={'TYPE': thermostat})
        md = Section('MD', subsections={'THERMOSTAT': thermostat}, keywords=md_keywords)
        self['MOTION'].insert(md)
    elif run_type == 'BAND':
        convergence_control_params = {'MAX_DR': Keyword('MAX_DR', max_drift), 'MAX_FORCE': Keyword('MAX_FORCE', max_force), 'RMS_DR': Keyword('RMS_DR', rms_drift), 'RMS_FORCE': Keyword('RMS_FORCE', rms_force)}
        band_kwargs = {'BAND_TYPE': Keyword('BAND_TYPE', 'IT-NEB', description='Improved tangent NEB'), 'NUMBER_OF_REPLICA': Keyword('NUMBER_OF_REPLICA'), 'NPROC_REP': Keyword('NPROC_REP', nproc_rep)}
        band = Section('BAND', keywords=band_kwargs)
        band.insert(Section('CONVERGENCE_CONTROL', keywords=convergence_control_params))
        self['MOTION'].insert(band)
    self.modify_dft_print_iters(0, add_last='numeric')
    if 'fix' in self.structure.site_properties:
        self['motion'].insert(Section('CONSTRAINT'))
        i = 0
        components = []
        tuples = []
        while i < len(self.structure):
            end = i + sum((1 for j in itertools.takewhile(lambda x: x == self.structure.site_properties['fix'][i], self.structure.site_properties['fix'][i:])))
            components.append(self.structure.site_properties['fix'][i])
            tuples.append((i + 1, end))
            i = end
        self['motion']['constraint'].insert(SectionList(sections=[Section('FIXED_ATOMS', keywords={'COMPONENTS_TO_FIX': Keyword('COMPONENTS_TO_FIX', c), 'LIST': Keyword('LIST', f'{t[0]}..{t[1]}')}) for t, c in zip(tuples, components) if c]))