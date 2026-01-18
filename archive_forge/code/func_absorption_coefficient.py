from __future__ import annotations
import os
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.constants import physical_constants, speed_of_light
from scipy.integrate import simps
from scipy.interpolate import interp1d
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.due import Doi, due
def absorption_coefficient(dielectric):
    """
    Calculate the optical absorption coefficient from an input set of
    pymatgen vasprun dielectric constant data.

    Args:
        dielectric (list): A list containing the dielectric response function
            in the pymatgen vasprun format.
            - element 0: list of energies
            - element 1: real dielectric tensors, in ``[xx, yy, zz, xy, xz, yz]`` format.
            - element 2: imaginary dielectric tensors, in ``[xx, yy, zz, xy, xz, yz]`` format.

    Returns:
        np.array: absorption coefficient using eV as frequency units (cm^-1).
    """
    energies_in_eV = np.array(dielectric[0])
    real_dielectric = parse_dielectric_data(dielectric[1])
    imag_dielectric = parse_dielectric_data(dielectric[2])
    epsilon_1 = np.mean(real_dielectric, axis=1)
    epsilon_2 = np.mean(imag_dielectric, axis=1)
    return (energies_in_eV, 2.0 * np.sqrt(2.0) * pi * eV_to_recip_cm * energies_in_eV * np.sqrt(-epsilon_1 + np.sqrt(epsilon_1 ** 2 + epsilon_2 ** 2)))