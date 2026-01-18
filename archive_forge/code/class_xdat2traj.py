import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
class xdat2traj:

    def __init__(self, trajectory=None, atoms=None, poscar=None, xdatcar=None, sort=None, calc=None):
        """
        trajectory is the name of the file to write the trajectory to
        poscar is the name of the poscar file to read. Default: POSCAR
        """
        if not poscar:
            self.poscar = 'POSCAR'
        else:
            self.poscar = poscar
        if not atoms:
            self.atoms = ase.io.read(self.poscar, format='vasp')
            resort_reqd = True
        else:
            self.atoms = atoms
            resort_reqd = False
        if not calc:
            self.calc = Vasp()
        else:
            self.calc = calc
        if not sort:
            if not hasattr(self.calc, 'sort'):
                self.calc.sort = list(range(len(self.atoms)))
        else:
            self.calc.sort = sort
        self.calc.resort = list(range(len(self.calc.sort)))
        for n in range(len(self.calc.resort)):
            self.calc.resort[self.calc.sort[n]] = n
        if not xdatcar:
            self.xdatcar = 'XDATCAR'
        else:
            self.xdatcar = xdatcar
        if not trajectory:
            self.trajectory = 'out.traj'
        else:
            self.trajectory = trajectory
        self.out = ase.io.trajectory.Trajectory(self.trajectory, mode='w')
        if resort_reqd:
            self.atoms = self.atoms[self.calc.resort]
        self.energies = self.calc.read_energy(all=True)[1]
        self.forces = self.calc.read_forces(self.atoms, all=True)

    def convert(self):
        lines = open(self.xdatcar).readlines()
        if len(lines[7].split()) == 0:
            del lines[0:8]
        elif len(lines[5].split()) == 0:
            del lines[0:6]
        elif len(lines[4].split()) == 0:
            del lines[0:5]
        elif lines[7].split()[0] == 'Direct':
            del lines[0:8]
        step = 0
        iatom = 0
        scaled_pos = []
        for line in lines:
            if iatom == len(self.atoms):
                if step == 0:
                    self.out.write_header(self.atoms[self.calc.resort])
                scaled_pos = np.array(scaled_pos)
                self.atoms.set_scaled_positions(scaled_pos[self.calc.resort])
                calc = SinglePointCalculator(self.atoms, energy=self.energies[step], forces=self.forces[step])
                self.atoms.calc = calc
                self.out.write(self.atoms)
                scaled_pos = []
                iatom = 0
                step += 1
            elif not line.split()[0] == 'Direct':
                iatom += 1
                scaled_pos.append([float(line.split()[n]) for n in range(3)])
        if step == 0:
            self.out.write_header(self.atoms[self.calc.resort])
        scaled_pos = np.array(scaled_pos)[self.calc.resort]
        self.atoms.set_scaled_positions(scaled_pos)
        calc = SinglePointCalculator(self.atoms, energy=self.energies[step], forces=self.forces[step])
        self.atoms.calc = calc
        self.out.write(self.atoms)
        self.out.close()