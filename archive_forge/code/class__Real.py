import torch
class _Real(Constraint):
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """

    def check(self, value):
        return value == value