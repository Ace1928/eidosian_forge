from pyomo.core.base import Transformation
class IsomorphicTransformation(Transformation):
    """
    Base class for 'lossless' transformations for which a bijective
    mapping between optimal variable values and the optimal cost
    exists.
    """

    def __init__(self, **kwds):
        kwds['name'] = kwds.get('name', 'isomorphic_transformation')
        super(IsomorphicTransformation, self).__init__(**kwds)