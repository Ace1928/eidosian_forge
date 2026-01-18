from pyomo.core.base import Transformation
class NonIsomorphicTransformation(Transformation):
    """
    Base class for 'lossy' transformations for which a bijective
    mapping between optimal variable values and the optimal cost does
    not  exist.
    """

    def __init__(self, **kwds):
        kwds['name'] = kwds.get('name', 'isomorphic_transformation')
        super(NonIsomorphicTransformation, self).__init__(**kwds)