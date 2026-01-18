from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class TetrahedralNonorientableCuspedCensus(PlatonicManifoldTable):
    """
        Iterator for the tetrahedral non-orientable cusped hyperbolic manifolds up to
        21 tetrahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic tetrahedra.

        >>> len(TetrahedralNonorientableCuspedCensus)
        25194
        >>> list(TetrahedralNonorientableCuspedCensus[:1.3])
        [ntet01_00000(0,0)]

        """
    _regex = re.compile('ntet\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'tetrahedral_nonorientable_cusped_census', **kwargs)