from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class CubicalOrientableCuspedCensus(PlatonicManifoldTable):
    """
        Iterator for the cubical orientable cusped hyperbolic manifolds up to
        7 cubes, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic octahedra.

        >>> M = TetrahedralOrientableCuspedCensus['otet05_00001']
        >>> CubicalOrientableCuspedCensus.identify(M)
        ocube01_00002(0,0)(0,0)

        """
    _regex = re.compile('ocube\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'cubical_orientable_cusped_census', **kwargs)