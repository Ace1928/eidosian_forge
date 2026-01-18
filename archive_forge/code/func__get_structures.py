from __future__ import annotations
import fractions
import itertools
import logging
import math
import re
import subprocess
from glob import glob
from shutil import which
from threading import Timer
import numpy as np
from monty.dev import requires
from monty.fractions import lcm
from monty.tempfile import ScratchDir
from pymatgen.core import DummySpecies, PeriodicSite, Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def _get_structures(self, num_structs):
    structs = []
    if '.py' in makestr_cmd:
        options = ['-input', 'struct_enum.out', str(1), str(num_structs)]
    else:
        options = ['struct_enum.out', str(0), str(num_structs - 1)]
    with subprocess.Popen([makestr_cmd, *options], stdout=subprocess.PIPE, stdin=subprocess.PIPE, close_fds=True) as rs:
        _stdout, stderr = rs.communicate()
    if stderr:
        logger.warning(stderr.decode())
    disordered_site_properties = {}
    if len(self.ordered_sites) > 0:
        original_latt = self.ordered_sites[0].lattice
        site_properties = {}
        for site in self.ordered_sites:
            for k, v in site.properties.items():
                disordered_site_properties[k] = None
                if k in site_properties:
                    site_properties[k].append(v)
                else:
                    site_properties[k] = [v]
        ordered_structure = Structure(original_latt, [site.species for site in self.ordered_sites], [site.frac_coords for site in self.ordered_sites], site_properties=site_properties)
        inv_org_latt = np.linalg.inv(original_latt.matrix)
    else:
        ordered_structure = None
        inv_org_latt = None
    for file in glob('vasp.*'):
        with open(file) as file:
            data = file.read()
            data = re.sub('scale factor', '1', data)
            data = re.sub('(\\d+)-(\\d+)', '\\1 -\\2', data)
            poscar = Poscar.from_str(data, self.index_species)
            sub_structure = poscar.structure
            new_latt = sub_structure.lattice
            sites = []
            if len(self.ordered_sites) > 0:
                transformation = np.dot(new_latt.matrix, inv_org_latt)
                transformation = [[int(round(cell)) for cell in row] for row in transformation]
                logger.debug(f'Supercell matrix: {transformation}')
                struct = ordered_structure * transformation
                sites.extend([site.to_unit_cell() for site in struct])
                super_latt = sites[-1].lattice
            else:
                super_latt = new_latt
            for site in sub_structure:
                if site.specie.symbol != 'X':
                    sites.append(PeriodicSite(site.species, site.frac_coords, super_latt, to_unit_cell=True, properties=disordered_site_properties))
                else:
                    logger.debug('Skipping sites that include species X.')
            structs.append(Structure.from_sites(sorted(sites)))
    logger.debug(f'Read in a total of {num_structs} structures.')
    return structs