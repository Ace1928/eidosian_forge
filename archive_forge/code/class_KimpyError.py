from ase.calculators.calculator import CalculatorError
class KimpyError(CalculatorError):
    """
    A call to a kimpy function resulted in a RuntimeError being raised
    """
    pass