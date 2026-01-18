import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
class AimsCube:
    """Object to ensure the output of cube files, can be attached to Aims object"""

    def __init__(self, origin=(0, 0, 0), edges=[(0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)], points=(50, 50, 50), plots=None):
        """parameters:

        origin, edges, points:
            Same as in the FHI-aims output
        plots:
            what to print, same names as in FHI-aims """
        self.name = 'AimsCube'
        self.origin = origin
        self.edges = edges
        self.points = points
        self.plots = plots

    def ncubes(self):
        """returns the number of cube files to output """
        if self.plots:
            number = len(self.plots)
        else:
            number = 0
        return number

    def set(self, **kwargs):
        """ set any of the parameters ... """

    def move_to_base_name(self, basename):
        """ when output tracking is on or the base namem is not standard,
        this routine will rename add the base to the cube file output for
        easier tracking """
        for plot in self.plots:
            found = False
            cube = plot.split()
            if cube[0] == 'total_density' or cube[0] == 'spin_density' or cube[0] == 'delta_density':
                found = True
                old_name = cube[0] + '.cube'
                new_name = basename + '.' + old_name
            if cube[0] == 'eigenstate' or cube[0] == 'eigenstate_density':
                found = True
                state = int(cube[1])
                s_state = cube[1]
                for i in [10, 100, 1000, 10000]:
                    if state < i:
                        s_state = '0' + s_state
                old_name = cube[0] + '_' + s_state + '_spin_1.cube'
                new_name = basename + '.' + old_name
            if found:
                os.system('mv ' + old_name + ' ' + new_name)

    def add_plot(self, name):
        """ in case you forgot one ... """
        self.plots += [name]

    def write(self, file):
        """ write the necessary output to the already opened control.in """
        file.write('output cube ' + self.plots[0] + '\n')
        file.write('   cube origin ')
        for ival in self.origin:
            file.write(str(ival) + ' ')
        file.write('\n')
        for i in range(3):
            file.write('   cube edge ' + str(self.points[i]) + ' ')
            for ival in self.edges[i]:
                file.write(str(ival) + ' ')
            file.write('\n')
        if self.ncubes() > 1:
            for i in range(self.ncubes() - 1):
                file.write('output cube ' + self.plots[i + 1] + '\n')