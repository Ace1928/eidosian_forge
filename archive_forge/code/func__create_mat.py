from __future__ import annotations
import logging
import subprocess
from shutil import which
import pandas as pd
from monty.dev import requires
from monty.json import MSONable
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper
def _create_mat(self):
    structure = self.structure
    mat_name = self.mat_name
    magmoms = structure.site_properties['magmom']
    mat_id_dict = {}
    nmats = 0
    for key in self.unique_site_ids:
        spin_up, spin_down = (False, False)
        nmats += 1
        for site in key:
            m = magmoms[site]
            if m > 0:
                spin_up = True
            if m < 0:
                spin_down = True
        for site in key:
            m = magmoms[site]
            if spin_up and (not spin_down):
                mat_id_dict[site] = nmats
            if spin_down and (not spin_up):
                mat_id_dict[site] = nmats
            if spin_up and spin_down:
                m0 = magmoms[key[0]]
                if m > 0 and m0 > 0:
                    mat_id_dict[site] = nmats
                if m < 0 and m0 < 0:
                    mat_id_dict[site] = nmats
                if m > 0 > m0:
                    mat_id_dict[site] = nmats + 1
                if m < 0 < m0:
                    mat_id_dict[site] = nmats + 1
        if spin_up and spin_down:
            nmats += 1
    mat_file = [f'material:num-materials={nmats}']
    for key in self.unique_site_ids:
        i = self.unique_site_ids[key]
        for site in key:
            mat_id = mat_id_dict[site]
            m_magnitude = abs(magmoms[site])
            if magmoms[site] > 0:
                spin = 1
            if magmoms[site] < 0:
                spin = -1
            atom = structure[i].species.reduced_formula
            mat_file += [f'material[{mat_id}]:material-element={atom}']
            mat_file += [f'material[{mat_id}]:damping-constant=1.0', f'material[{mat_id}]:uniaxial-anisotropy-constant=1.0e-24', f'material[{mat_id}]:atomic-spin-moment={m_magnitude:.2f} !muB', f'material[{mat_id}]:initial-spin-direction=0,0,{spin}']
    mat_file = '\n'.join(mat_file)
    mat_file_name = mat_name + '.mat'
    self.mat_id_dict = mat_id_dict
    with open(mat_file_name, mode='w') as file:
        file.write(mat_file)