from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class IcosahedralNonorientableClosedCensus(PlatonicManifoldTable):
    """
        Iterator for the icosahedral non-orientable closed hyperbolic manifolds up
        to 3 icosahedra, i.e., manifolds that admit a tessellation by regular finite
        hyperbolic icosahedra.

        >>> list(IcosahedralNonorientableClosedCensus)
        [nicocld02_00000(1,0)]

        """
    _regex = re.compile('nicocld\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'icosahedral_nonorientable_closed_census', **kwargs)