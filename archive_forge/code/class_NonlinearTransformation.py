from pyomo.core.base import Transformation
class NonlinearTransformation(Transformation):
    """Base class for all nonlinear model transformations."""

    def __init__(self, **kwds):
        kwds['name'] = kwds.get('name', 'nonlinear_transform')
        super(NonlinearTransformation, self).__init__(**kwds)