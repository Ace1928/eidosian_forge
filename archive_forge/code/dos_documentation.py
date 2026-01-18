import numpy as np
import os
import subprocess
import warnings
from ase.calculators.openmx.reader import rn as read_nth_to_last_value

        Wraps all the density of states processing functions. Can go from
        .Dos.val and .Dos.vec files to a graphical figure showing many
        different PDOS plots against energy in one step.
        :param atom_index_list:
        :param method: method to be used to calculate the density of states
                       from eigenvalues and eigenvectors.
                       ('Tetrahedron' or 'Gaussian')
        :param gaussian_width: If the method is 'Gaussian' then gaussian_width
                               is required (eV).
        :param pdos: If True, the pseudo-density of states is calculated for a
                     given list of atoms for each orbital. If the system is
                     spin polarized, then each up and down state is also
                     calculated.
        :param orbital_list: If pdos is True, a list of atom indices are
                             required to generate the pdos of each of those
                             specified atoms.
        :param spin_polarization: If spin_polarization is False then spin
                                  states will not be separated in different
                                  PDOS's.
        :param density: If True, density of states will be plotted
        :param cum: If True, cumulative (or integrated) density of states will
                    be plotted
        :param erange: range of energies to view the DOS
        :param file_format: If not None, a file will be saved automatically in
                            that format ('pdf', 'png', 'jpeg' etc.)
        :return: matplotlib.figure.Figure object
        