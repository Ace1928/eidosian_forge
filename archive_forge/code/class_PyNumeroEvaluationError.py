class PyNumeroEvaluationError(ArithmeticError):
    """An exception to be raised by PyNumero evaluation backends in the event
    of a failed function evaluation. This should be caught by solver interfaces
    and translated to the solver-specific evaluation error API.

    """
    pass