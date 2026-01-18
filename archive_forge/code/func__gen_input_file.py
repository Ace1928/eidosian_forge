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
def _gen_input_file(self):
    """Generate the necessary struct_enum.in file for enumlib. See enumlib
        documentation for details.
        """
    coord_format = '{:.6f} {:.6f} {:.6f}'
    fitter = SpacegroupAnalyzer(self.structure, self.symm_prec)
    symmetrized_structure = fitter.get_symmetrized_structure()
    logger.debug(f'Spacegroup {fitter.get_space_group_symbol()} ({fitter.get_space_group_number()}) with {len(symmetrized_structure.equivalent_sites)} distinct sites')
    '\n        Enumlib doesn"t work when the number of species get too large. To\n        simplify matters, we generate the input file only with disordered sites\n        and exclude the ordered sites from the enumeration. The fact that\n        different disordered sites with the exact same species may belong to\n        different equivalent sites is dealt with by having determined the\n        spacegroup earlier and labelling the species differently.\n        '
    index_species = []
    index_amounts = []
    ordered_sites = []
    disordered_sites = []
    coord_str = []
    for sites in symmetrized_structure.equivalent_sites:
        if sites[0].is_ordered:
            ordered_sites.append(sites)
        else:
            sp_label = []
            species = dict(sites[0].species.items())
            if sum(species.values()) < 1 - EnumlibAdaptor.amount_tol:
                species[DummySpecies('X')] = 1 - sum(species.values())
            for sp, amt in species.items():
                if sp not in index_species:
                    index_species.append(sp)
                    sp_label.append(len(index_species) - 1)
                    index_amounts.append(amt * len(sites))
                else:
                    ind = index_species.index(sp)
                    sp_label.append(ind)
                    index_amounts[ind] += amt * len(sites)
            sp_label = '/'.join((f'{i}' for i in sorted(sp_label)))
            for site in sites:
                coord_str.append(f'{coord_format.format(*site.coords)} {sp_label}')
            disordered_sites.append(sites)

    def get_sg_info(ss):
        finder = SpacegroupAnalyzer(Structure.from_sites(ss), self.symm_prec)
        return finder.get_space_group_number()
    target_sg_num = get_sg_info(list(symmetrized_structure))
    curr_sites = list(itertools.chain.from_iterable(disordered_sites))
    sg_num = get_sg_info(curr_sites)
    ordered_sites = sorted(ordered_sites, key=len)
    logger.debug(f'Disordered sites has sg # {sg_num}')
    self.ordered_sites = []
    if self.check_ordered_symmetry:
        while sg_num != target_sg_num and len(ordered_sites) > 0:
            sites = ordered_sites.pop(0)
            temp_sites = list(curr_sites) + sites
            new_sg_num = get_sg_info(temp_sites)
            if sg_num != new_sg_num:
                logger.debug(f'Adding {sites[0].specie} in enum. New sg # {new_sg_num}')
                index_species.append(sites[0].specie)
                index_amounts.append(len(sites))
                sp_label = len(index_species) - 1
                for site in sites:
                    coord_str.append(f'{coord_format.format(*site.coords)} {sp_label}')
                disordered_sites.append(sites)
                curr_sites = temp_sites
                sg_num = new_sg_num
            else:
                self.ordered_sites.extend(sites)
    for sites in ordered_sites:
        self.ordered_sites.extend(sites)
    self.index_species = index_species
    lattice = self.structure.lattice
    output = [self.structure.formula, 'bulk']
    for vec in lattice.matrix:
        output.append(coord_format.format(*vec))
    output.extend((f'{len(index_species)}', f'{len(coord_str)}'))
    output.extend(coord_str)
    output.extend((f'{self.min_cell_size} {self.max_cell_size}', str(self.enum_precision_parameter), 'full'))
    n_disordered = sum((len(s) for s in disordered_sites))
    base = int(n_disordered * lcm(*(fraction.limit_denominator(n_disordered * self.max_cell_size).denominator for fraction in map(fractions.Fraction, index_amounts))))
    base *= 10
    total_amounts = sum(index_amounts)
    for amt in index_amounts:
        conc = amt / total_amounts
        if abs(conc * base - round(conc * base)) < 1e-05:
            output.append(f'{int(round(conc * base))} {int(round(conc * base))} {base}')
        else:
            min_conc = int(math.floor(conc * base))
            output.append(f'{min_conc - 1} {min_conc + 1} {base}')
    output.append('')
    logger.debug('Generated input file:\n' + '\n'.join(output))
    with open('struct_enum.in', mode='w') as file:
        file.write('\n'.join(output))