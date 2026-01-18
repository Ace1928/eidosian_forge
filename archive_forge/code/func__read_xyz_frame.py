from itertools import islice
import re
import warnings
from io import StringIO, UnsupportedOperation
import json
import numpy as np
import numbers
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from ase.utils import reader
def _read_xyz_frame(lines, natoms, properties_parser=key_val_str_to_dict, nvec=0):
    line = next(lines).strip()
    if nvec > 0:
        info = {'comment': line}
    else:
        info = properties_parser(line) if line else {}
    pbc = None
    if 'pbc' in info:
        pbc = info['pbc']
        del info['pbc']
    elif 'Lattice' in info:
        pbc = [True, True, True]
    elif nvec > 0:
        pbc = [False, False, False]
    cell = None
    if 'Lattice' in info:
        cell = info['Lattice'].T
        del info['Lattice']
    elif nvec > 0:
        cell = np.zeros((3, 3))
    if 'Properties' not in info:
        info['Properties'] = 'species:S:1:pos:R:3'
    properties, names, dtype, convs = parse_properties(info['Properties'])
    del info['Properties']
    data = []
    for ln in range(natoms):
        try:
            line = next(lines)
        except StopIteration:
            raise XYZError('ase.io.extxyz: Frame has {} atoms, expected {}'.format(len(data), natoms))
        vals = line.split()
        row = tuple([conv(val) for conv, val in zip(convs, vals)])
        data.append(row)
    try:
        data = np.array(data, dtype)
    except TypeError:
        raise XYZError('Badly formatted data or end of file reached before end of frame')
    if nvec > 0:
        for ln in range(nvec):
            try:
                line = next(lines)
            except StopIteration:
                raise XYZError('ase.io.adfxyz: Frame has {} cell vectors, expected {}'.format(len(cell), nvec))
            entry = line.split()
            if not entry[0].startswith('VEC'):
                raise XYZError('Expected cell vector, got {}'.format(entry[0]))
            try:
                n = int(entry[0][3:])
            except ValueError as e:
                raise XYZError('Expected VEC{}, got VEC{}'.format(ln + 1, entry[0][3:])) from e
            if n != ln + 1:
                raise XYZError('Expected VEC{}, got VEC{}'.format(ln + 1, n))
            cell[ln] = np.array([float(x) for x in entry[1:]])
            pbc[ln] = True
        if nvec != pbc.count(True):
            raise XYZError('Problem with number of cell vectors')
        pbc = tuple(pbc)
    arrays = {}
    for name in names:
        ase_name, cols = properties[name]
        if cols == 1:
            value = data[name]
        else:
            value = np.vstack([data[name + str(c)] for c in range(cols)]).T
        arrays[ase_name] = value
    symbols = None
    if 'symbols' in arrays:
        symbols = [s.capitalize() for s in arrays['symbols']]
        del arrays['symbols']
    numbers = None
    duplicate_numbers = None
    if 'numbers' in arrays:
        if symbols is None:
            numbers = arrays['numbers']
        else:
            duplicate_numbers = arrays['numbers']
        del arrays['numbers']
    charges = None
    if 'charges' in arrays:
        charges = arrays['charges']
        del arrays['charges']
    positions = None
    if 'positions' in arrays:
        positions = arrays['positions']
        del arrays['positions']
    atoms = Atoms(symbols=symbols, positions=positions, numbers=numbers, charges=charges, cell=cell, pbc=pbc, info=info)
    if 'move_mask' in arrays:
        if properties['move_mask'][1] == 3:
            cons = []
            for a in range(natoms):
                cons.append(FixCartesian(a, mask=~arrays['move_mask'][a, :]))
            atoms.set_constraint(cons)
        elif properties['move_mask'][1] == 1:
            atoms.set_constraint(FixAtoms(mask=~arrays['move_mask']))
        else:
            raise XYZError('Not implemented constraint')
        del arrays['move_mask']
    for name, array in arrays.items():
        atoms.new_array(name, array)
    if duplicate_numbers is not None:
        atoms.set_atomic_numbers(duplicate_numbers)
    results = {}
    for key in list(atoms.info.keys()):
        if key in per_config_properties:
            results[key] = atoms.info[key]
            if key == 'stress' and results[key].shape == (3, 3):
                stress = results[key]
                stress = np.array([stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2], stress[0, 2], stress[0, 1]])
                results[key] = stress
    for key in list(atoms.arrays.keys()):
        if key in per_atom_properties and len(value.shape) >= 1 and (value.shape[0] == len(atoms)):
            results[key] = atoms.arrays[key]
    if results != {}:
        calculator = SinglePointCalculator(atoms, **results)
        atoms.calc = calculator
    return atoms