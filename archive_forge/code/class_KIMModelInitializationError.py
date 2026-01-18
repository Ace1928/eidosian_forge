from ase.calculators.calculator import CalculatorError
class KIMModelInitializationError(CalculatorError):
    """
    KIM API Model object or ComputeArguments object could not be
    successfully created
    """
    pass