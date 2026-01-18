from __future__ import annotations
import logging
import subprocess
from shutil import which
import pandas as pd
from monty.dev import requires
from monty.json import MSONable
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper
def _create_ucf(self):
    structure = self.structure
    mat_name = self.mat_name
    abc = structure.lattice.abc
    ucx, ucy, ucz = (abc[0], abc[1], abc[2])
    ucf = ['# Unit cell size:']
    ucf += [f'{ucx:.10f} {ucy:.10f} {ucz:.10f}']
    ucf += ['# Unit cell lattice vectors:']
    a1 = list(structure.lattice.matrix[0])
    ucf += [f'{a1[0]:.10f} {a1[1]:.10f} {a1[2]:.10f}']
    a2 = list(structure.lattice.matrix[1])
    ucf += [f'{a2[0]:.10f} {a2[1]:.10f} {a2[2]:.10f}']
    a3 = list(structure.lattice.matrix[2])
    ucf += [f'{a3[0]:.10f} {a3[1]:.10f} {a3[2]:.10f}']
    nmats = max(self.mat_id_dict.values())
    ucf += ['# Atoms num_materials; id cx cy cz mat cat hcat']
    ucf += [f'{len(structure)} {nmats}']
    for site, r in enumerate(structure.frac_coords):
        mat_id = self.mat_id_dict[site] - 1
        ucf += [f'{site} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f} {mat_id} 0 0']
    sgraph = self.sgraph
    n_inter = 0
    for idx in range(len(sgraph.graph.nodes)):
        n_inter += sgraph.get_coordination_of_site(idx)
    ucf += ['# Interactions']
    ucf += [f'{n_inter} isotropic']
    iid = 0
    for idx in range(len(sgraph.graph.nodes)):
        connections = sgraph.get_connected_sites(idx)
        for c in connections:
            jimage = c[1]
            dx = jimage[0]
            dy = jimage[1]
            dz = jimage[2]
            j = c[2]
            dist = round(c[-1], 2)
            j_exc = self.hm.javg if self.avg is True else self.hm._get_j_exc(idx, j, dist)
            j_exc *= 1.6021766e-22
            j_exc = str(j_exc)
            ucf += [f'{iid} {idx} {j} {dx} {dy} {dz} {j_exc}']
            iid += 1
    ucf = '\n'.join(ucf)
    ucf_file_name = mat_name + '.ucf'
    with open(ucf_file_name, mode='w') as file:
        file.write(ucf)