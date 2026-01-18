from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def _build_dataset_dict(spg_ds):
    keys = ('number', 'hall_number', 'international', 'hall', 'choice', 'transformation_matrix', 'origin_shift', 'rotations', 'translations', 'wyckoffs', 'site_symmetry_symbols', 'crystallographic_orbits', 'equivalent_atoms', 'primitive_lattice', 'mapping_to_primitive', 'std_lattice', 'std_types', 'std_positions', 'std_rotation_matrix', 'std_mapping_to_primitive', 'pointgroup')
    dataset = {}
    for key, data in zip(keys, spg_ds):
        dataset[key] = data
    dataset['international'] = dataset['international'].strip()
    dataset['hall'] = dataset['hall'].strip()
    dataset['choice'] = dataset['choice'].strip()
    dataset['transformation_matrix'] = np.array(dataset['transformation_matrix'], dtype='double', order='C')
    dataset['origin_shift'] = np.array(dataset['origin_shift'], dtype='double')
    dataset['rotations'] = np.array(dataset['rotations'], dtype='intc', order='C')
    dataset['translations'] = np.array(dataset['translations'], dtype='double', order='C')
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    dataset['wyckoffs'] = [letters[x] for x in dataset['wyckoffs']]
    dataset['site_symmetry_symbols'] = [s.strip() for s in dataset['site_symmetry_symbols']]
    dataset['crystallographic_orbits'] = np.array(dataset['crystallographic_orbits'], dtype='intc')
    dataset['equivalent_atoms'] = np.array(dataset['equivalent_atoms'], dtype='intc')
    dataset['primitive_lattice'] = np.array(np.transpose(dataset['primitive_lattice']), dtype='double', order='C')
    dataset['mapping_to_primitive'] = np.array(dataset['mapping_to_primitive'], dtype='intc')
    dataset['std_lattice'] = np.array(np.transpose(dataset['std_lattice']), dtype='double', order='C')
    dataset['std_types'] = np.array(dataset['std_types'], dtype='intc')
    dataset['std_positions'] = np.array(dataset['std_positions'], dtype='double', order='C')
    dataset['std_rotation_matrix'] = np.array(dataset['std_rotation_matrix'], dtype='double', order='C')
    dataset['std_mapping_to_primitive'] = np.array(dataset['std_mapping_to_primitive'], dtype='intc')
    dataset['pointgroup'] = dataset['pointgroup'].strip()
    return dataset