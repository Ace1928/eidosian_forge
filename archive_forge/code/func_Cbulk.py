from pytest import approx, fixture
from ase import Atoms
from ase.build import bulk
from ase.vibrations.raman import StaticRamanCalculator
from ase.vibrations.raman import StaticRamanPhononsCalculator
from ase.vibrations.placzek import PlaczekStatic
from ase.vibrations.placzek import PlaczekStaticPhonons
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.emt import EMT
@fixture(scope='module')
def Cbulk():
    Cbulk = bulk('C', crystalstructure='fcc', a=2 * 1.221791471)
    Cbulk = Cbulk.repeat([2, 1, 1])
    Cbulk.calc = EMT()
    return Cbulk