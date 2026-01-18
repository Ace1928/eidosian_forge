import os
import re
import numpy as np
from ase.units import eV, Ang
from ase.calculators.calculator import FileIOCalculator, ReadError
class GULP(FileIOCalculator):
    implemented_properties = ['energy', 'forces', 'stress']
    command = 'gulp < PREFIX.gin > PREFIX.got'
    discard_results_on_any_change = True
    default_parameters = dict(keywords='conp gradients', options=[], shel=[], library='ffsioh.lib', conditions=None)

    def get_optimizer(self, atoms):
        gulp_keywords = self.parameters.keywords.split()
        if 'opti' not in gulp_keywords:
            raise ValueError('Can only create optimizer from GULP calculator with "opti" keyword.  Current keywords: {}'.format(gulp_keywords))
        opt = GULPOptimizer(atoms, self)
        return opt

    def __init__(self, restart=None, ignore_bad_restart_file=FileIOCalculator._deprecated, label='gulp', atoms=None, optimized=None, Gnorm=1000.0, steps=1000, conditions=None, **kwargs):
        """Construct GULP-calculator object."""
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file, label, atoms, **kwargs)
        self.optimized = optimized
        self.Gnorm = Gnorm
        self.steps = steps
        self.conditions = conditions
        self.library_check()
        self.atom_types = []
        self.fractional_coordinates = None

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = self.parameters
        s = p.keywords
        s += '\ntitle\nASE calculation\nend\n\n'
        if all(self.atoms.pbc):
            cell_params = self.atoms.cell.cellpar()
            s += 'cell\n{0:9.6f} {1:9.6f} {2:9.6f} {3:8.5f} {4:8.5f} {5:8.5f}\n'.format(*cell_params)
            s += 'frac\n'
            coords = self.atoms.get_scaled_positions()
        else:
            s += 'cart\n'
            coords = self.atoms.get_positions()
        if self.conditions is not None:
            c = self.conditions
            labels = c.get_atoms_labels()
            self.atom_types = c.get_atom_types()
        else:
            labels = self.atoms.get_chemical_symbols()
        for xyz, symbol in zip(coords, labels):
            s += ' {0:2} core {1:10.7f}  {2:10.7f}  {3:10.7f}\n'.format(symbol, *xyz)
            if symbol in p.shel:
                s += ' {0:2} shel {1:10.7f}  {2:10.7f}  {3:10.7f}\n'.format(symbol, *xyz)
        s += '\nlibrary {0}\n'.format(p.library)
        if p.options:
            for t in p.options:
                s += '%s\n' % t
        with open(self.prefix + '.gin', 'w') as fd:
            fd.write(s)

    def read_results(self):
        FileIOCalculator.read(self, self.label)
        if not os.path.isfile(self.label + '.got'):
            raise ReadError
        with open(self.label + '.got') as fd:
            lines = fd.readlines()
        cycles = -1
        self.optimized = None
        for i, line in enumerate(lines):
            m = re.match('\\s*Total lattice energy\\s*=\\s*(\\S+)\\s*eV', line)
            if m:
                energy = float(m.group(1))
                self.results['energy'] = energy
                self.results['free_energy'] = energy
            elif line.find('Optimisation achieved') != -1:
                self.optimized = True
            elif line.find('Final Gnorm') != -1:
                self.Gnorm = float(line.split()[-1])
            elif line.find('Cycle:') != -1:
                cycles += 1
            elif line.find('Final Cartesian derivatives') != -1:
                s = i + 5
                forces = []
                while True:
                    s = s + 1
                    if lines[s].find('------------') != -1:
                        break
                    if lines[s].find(' s ') != -1:
                        continue
                    g = lines[s].split()[3:6]
                    G = [-float(x) * eV / Ang for x in g]
                    forces.append(G)
                forces = np.array(forces)
                self.results['forces'] = forces
            elif line.find('Final internal derivatives') != -1:
                s = i + 5
                forces = []
                while True:
                    s = s + 1
                    if lines[s].find('------------') != -1:
                        break
                    g = lines[s].split()[3:6]
                    "for t in range(3-len(g)):\n                        g.append(' ')\n                    for j in range(2):\n                        min_index=[i+1 for i,e in enumerate(g[j][1:]) if e == '-']\n                        if j==0 and len(min_index) != 0:\n                            if len(min_index)==1:\n                                g[2]=g[1]\n                                g[1]=g[0][min_index[0]:]\n                                g[0]=g[0][:min_index[0]]\n                            else:\n                                g[2]=g[0][min_index[1]:]\n                                g[1]=g[0][min_index[0]:min_index[1]]\n                                g[0]=g[0][:min_index[0]]\n                                break\n                        if j==1 and len(min_index) != 0:\n                            g[2]=g[1][min_index[0]:]\n                            g[1]=g[1][:min_index[0]]"
                    G = [-float(x) * eV / Ang for x in g]
                    forces.append(G)
                forces = np.array(forces)
                self.results['forces'] = forces
            elif line.find('Final cartesian coordinates of atoms') != -1:
                s = i + 5
                positions = []
                while True:
                    s = s + 1
                    if lines[s].find('------------') != -1:
                        break
                    if lines[s].find(' s ') != -1:
                        continue
                    xyz = lines[s].split()[3:6]
                    XYZ = [float(x) * Ang for x in xyz]
                    positions.append(XYZ)
                positions = np.array(positions)
                self.atoms.set_positions(positions)
            elif line.find('Final stress tensor components') != -1:
                res = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for j in range(3):
                    var = lines[i + j + 3].split()[1]
                    res[j] = float(var)
                    var = lines[i + j + 3].split()[3]
                    res[j + 3] = float(var)
                stress = np.array(res)
                self.results['stress'] = stress
            elif line.find('Final Cartesian lattice vectors') != -1:
                lattice_vectors = np.zeros((3, 3))
                s = i + 2
                for j in range(s, s + 3):
                    temp = lines[j].split()
                    for k in range(3):
                        lattice_vectors[j - s][k] = float(temp[k])
                self.atoms.set_cell(lattice_vectors)
                if self.fractional_coordinates is not None:
                    self.fractional_coordinates = np.array(self.fractional_coordinates)
                    self.atoms.set_scaled_positions(self.fractional_coordinates)
            elif line.find('Final fractional coordinates of atoms') != -1:
                s = i + 5
                scaled_positions = []
                while True:
                    s = s + 1
                    if lines[s].find('------------') != -1:
                        break
                    if lines[s].find(' s ') != -1:
                        continue
                    xyz = lines[s].split()[3:6]
                    XYZ = [float(x) for x in xyz]
                    scaled_positions.append(XYZ)
                self.fractional_coordinates = scaled_positions
        self.steps = cycles

    def get_opt_state(self):
        return self.optimized

    def get_opt_steps(self):
        return self.steps

    def get_Gnorm(self):
        return self.Gnorm

    def library_check(self):
        if self.parameters['library'] is not None:
            if 'GULP_LIB' not in os.environ:
                raise RuntimeError('Be sure to have set correctly $GULP_LIB or to have the force field library.')