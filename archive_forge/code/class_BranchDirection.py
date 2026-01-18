from pyomo.common import Factory
class BranchDirection(object):
    """Allowed values for MIP variable branching directions in the `direction` Suffix of a model."""
    default = 0
    down = -1
    up = 1
    ALL = {default, down, up}