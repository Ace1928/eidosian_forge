class PowerIterationFailedConvergence(ExceededMaxIterations):
    """Raised when the power iteration method fails to converge within a
    specified iteration limit.

    `num_iterations` is the number of iterations that have been
    completed when this exception was raised.

    """

    def __init__(self, num_iterations, *args, **kw):
        msg = f'power iteration failed to converge within {num_iterations} iterations'
        exception_message = msg
        superinit = super().__init__
        superinit(self, exception_message, *args, **kw)