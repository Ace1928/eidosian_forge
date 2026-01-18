import torch
class _IntegerLessThan(Constraint):
    """
    Constrain to an integer interval `(-inf, upper_bound]`.
    """
    is_discrete = True

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (value % 1 == 0) & (value <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f'(upper_bound={self.upper_bound})'
        return fmt_string