from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class TetrahedralOrientableCuspedCensus(PlatonicManifoldTable):
    """
        Iterator for the tetrahedral orientable cusped hyperbolic manifolds up to
        25 tetrahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic tetrahedra.

        >>> for M in TetrahedralOrientableCuspedCensus(solids = 5): # doctest: +NUMERIC6
        ...     print(M, M.volume())
        otet05_00000(0,0) 5.07470803
        otet05_00001(0,0)(0,0) 5.07470803
        >>> TetrahedralOrientableCuspedCensus.identify(Manifold("m004"))
        otet02_00001(0,0)


        """
    _regex = re.compile('otet\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, table='tetrahedral_orientable_cusped_census', **kwargs)