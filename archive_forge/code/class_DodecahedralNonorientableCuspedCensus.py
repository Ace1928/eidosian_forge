from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class DodecahedralNonorientableCuspedCensus(PlatonicManifoldTable):
    """
        Iterator for the dodecahedral non-orientable cusped hyperbolic manifolds up to
        2 dodecahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic dodecahedra.

        >>> len(DodecahedralNonorientableCuspedCensus)
        4146

        """
    _regex = re.compile('ndode\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'dodecahedral_nonorientable_cusped_census', **kwargs)