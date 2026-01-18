import numpy as np
from ase.units import Hartree, Bohr
class Excitation:
    """Base class for a single excitation"""

    def __init__(self, energy, index, mur, muv=None, magn=None):
        """
        Parameters
        ----------
        energy: float
          Energy realtive to the ground state
        index: int
          Excited state index
        mur: list of three floats or complex numbers
          Length form dipole matrix element
        muv: list of three floats or complex numbers or None
          Velocity form dipole matrix element, default None
        magn: list of three floats or complex numbers or None
          Magnetic matrix element, default None
        """
        self.energy = energy
        self.index = index
        self.mur = mur
        self.muv = muv
        self.magn = magn
        self.fij = 1.0

    def outstring(self):
        """Format yourself as a string"""
        string = '{0:g}  {1}  '.format(self.energy, self.index)

        def format_me(me):
            string = ''
            if me.dtype == float:
                for m in me:
                    string += ' {0:g}'.format(m)
            else:
                for m in me:
                    string += ' {0.real:g}{0.imag:+g}j'.format(m)
            return string
        string += '  ' + format_me(self.mur)
        if self.muv is not None:
            string += '  ' + format_me(self.muv)
        if self.magn is not None:
            string += '  ' + format_me(self.magn)
        string += '\n'
        return string

    @classmethod
    def fromstring(cls, string):
        """Initialize yourself from a string"""
        l = string.split()
        energy = float(l.pop(0))
        index = int(l.pop(0))
        mur = np.array([float(l.pop(0)) for i in range(3)])
        try:
            muv = np.array([float(l.pop(0)) for i in range(3)])
        except IndexError:
            muv = None
        try:
            magn = np.array([float(l.pop(0)) for i in range(3)])
        except IndexError:
            magn = None
        return cls(energy, index, mur, muv, magn)

    def get_dipole_me(self, form='r'):
        """Return the excitations dipole matrix element
        including the occupation factor sqrt(fij)"""
        if form == 'r':
            me = -self.mur
        elif form == 'v':
            me = -self.muv
        else:
            raise RuntimeError('Unknown form >' + form + '<')
        return np.sqrt(self.fij) * me

    def get_dipole_tensor(self, form='r'):
        """Return the oscillator strength tensor"""
        me = self.get_dipole_me(form)
        return 2 * np.outer(me, me.conj()) * self.energy / Hartree

    def get_oscillator_strength(self, form='r'):
        """Return the excitations dipole oscillator strength."""
        me2_c = self.get_dipole_tensor(form).diagonal().real
        return np.array([np.sum(me2_c) / 3.0] + me2_c.tolist())