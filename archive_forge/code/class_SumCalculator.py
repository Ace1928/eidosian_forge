from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
class SumCalculator(LinearCombinationCalculator):
    """SumCalculator for combining multiple calculators.

    This calculator can be used when there are different calculators for the different chemical environment or
    for example during delta leaning. It works with a list of arbitrary calculators and evaluates them in sequence
    when it is required.
    The supported properties are the intersection of the implemented properties in each calculator.
    """

    def __init__(self, calcs, atoms=None):
        """Implementation of sum of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
        """
        weights = [1.0] * len(calcs)
        super().__init__(calcs, weights, atoms)