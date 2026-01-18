from itertools import count
import numpy as np
from ase import Atoms
from ase.units import invcm, Ha
from ase.data import atomic_masses
from ase.calculators.calculator import all_changes
from ase.calculators.morse import MorsePotential
from ase.calculators.excitation_list import Excitation, ExcitationList
class H2MorseCalculator(MorsePotential):
    """H2 ground or excited state as Morse potential"""
    _count = count(0)

    def __init__(self, restart=None, state=0, rng=np.random):
        self.rng = rng
        MorsePotential.__init__(self, restart=restart, epsilon=De[state], r0=Re[state], rho0=rho0[state])

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        if atoms is not None:
            assert len(atoms) == 2
        MorsePotential.calculate(self, atoms, properties, system_changes)
        vr = atoms[1].position - atoms[0].position
        r = np.linalg.norm(vr)
        hr = vr / r
        vrand = self.rng.rand(3)
        hx = np.cross(hr, vrand)
        hx /= np.linalg.norm(hx)
        hy = np.cross(hr, hx)
        hy /= np.linalg.norm(hy)
        wfs = [1, hr, hx, hy]
        berry = (-1) ** self.rng.randint(0, 2, 4)
        self.wfs = [wf * b for wf, b in zip(wfs, berry)]

    def read(self, filename):
        ms = self
        with open(filename) as fd:
            ms.wfs = [int(fd.readline().split()[0])]
            for i in range(1, 4):
                ms.wfs.append(np.array([float(x) for x in fd.readline().split()[:4]]))
        ms.filename = filename
        return ms

    def write(self, filename, option=None):
        """write calculated state to a file"""
        with open(filename, 'w') as fd:
            fd.write('{}\n'.format(self.wfs[0]))
            for wf in self.wfs[1:]:
                fd.write('{0:g} {1:g} {2:g}\n'.format(*wf))

    def overlap(self, other):
        ov = np.zeros((4, 4))
        ov[0, 0] = self.wfs[0] * other.wfs[0]
        wfs = np.array(self.wfs[1:])
        owfs = np.array(other.wfs[1:])
        ov[1:, 1:] = np.dot(wfs, owfs.T)
        return ov