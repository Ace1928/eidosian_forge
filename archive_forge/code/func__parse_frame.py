from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import reader
import re
import xml.etree.ElementTree as ET
def _parse_frame(tree, species):
    """Parse a certain frame from QBOX output

    Inputs:
        tree - ElementTree, <iteration> block from output file
        species - dict, data about species. Key is name of atom type,
            value is data about that type
    Return:
        Atoms object describing this iteration"""
    energy = float(tree.find('etotal').text)
    unitcell = tree.find('atomset').find('unit_cell')
    cell = []
    for d in ['a', 'b', 'c']:
        cell.append([float(x) for x in unitcell.get(d).split()])
    stress_tree = tree.find('stress_tensor')
    if stress_tree is None:
        stresses = None
    else:
        stresses = [float(stress_tree.find('sigma_%s' % x).text) for x in ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']]
    atoms = Atoms(pbc=True, cell=cell)
    forces = []
    for atom in tree.find('atomset').findall('atom'):
        spec = atom.get('species')
        symbol = species[spec]['symbol']
        mass = species[spec]['mass']
        pos = [float(x) for x in atom.find('position').text.split()]
        force = [float(x) for x in atom.find('force').text.split()]
        momentum = [float(x) * mass for x in atom.find('velocity').text.split()]
        atom = Atom(symbol=symbol, mass=mass, position=pos, momentum=momentum)
        atoms += atom
        forces.append(force)
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stresses)
    atoms.calc = calc
    return atoms