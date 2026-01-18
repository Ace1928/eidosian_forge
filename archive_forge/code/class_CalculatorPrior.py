import numpy as np
class CalculatorPrior(Prior):
    """CalculatorPrior object, allows the user to
    use another calculator as prior function instead of the
    default constant.

    Parameters:

    atoms: the Atoms object
    calculator: one of ASE's calculators
    """

    def __init__(self, atoms, calculator):
        Prior.__init__(self)
        self.atoms = atoms.copy()
        self.atoms.calc = calculator

    def potential(self, x):
        self.atoms.set_positions(x.reshape(-1, 3))
        V = self.atoms.get_potential_energy(force_consistent=True)
        gradV = -self.atoms.get_forces().reshape(-1)
        return np.append(np.array(V).reshape(-1), gradV)