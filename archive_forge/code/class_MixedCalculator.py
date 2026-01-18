from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
class MixedCalculator(LinearCombinationCalculator):
    """
    Mixing of two calculators with different weights

    H = weight1 * H1 + weight2 * H2

    Has functionality to get the energy contributions from each calculator

    Parameters
    ----------
    calc1 : ASE-calculator
    calc2 : ASE-calculator
    weight1 : float
        weight for calculator 1
    weight2 : float
        weight for calculator 2
    """

    def __init__(self, calc1, calc2, weight1, weight2):
        super().__init__([calc1, calc2], [weight1, weight2])

    def set_weights(self, w1, w2):
        self.weights[0] = w1
        self.weights[1] = w2

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """ Calculates all the specific property for each calculator and returns with the summed value.
        """
        super().calculate(atoms, properties, system_changes)
        if 'energy' in properties:
            energy1 = self.calcs[0].get_property('energy', atoms)
            energy2 = self.calcs[1].get_property('energy', atoms)
            self.results['energy_contributions'] = (energy1, energy2)

    def get_energy_contributions(self, atoms=None):
        """ Return the potential energy from calc1 and calc2 respectively """
        self.calculate(properties=['energy'], atoms=atoms)
        return self.results['energy_contributions']