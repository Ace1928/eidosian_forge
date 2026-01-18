from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class NonorientableCuspedCensus(ManifoldTable):
    """
        Iterator for all orientable cusped hyperbolic manifolds that
        can be triangulated with at most 5 ideal tetrahedra.

        >>> for M in NonorientableCuspedCensus(betti=2)[:3]:
        ...   print(M, M.homology())
        ... 
        m124(0,0)(0,0)(0,0) Z/2 + Z + Z
        m128(0,0)(0,0) Z + Z
        m131(0,0) Z + Z
        """
    _regex = re.compile('[mxy]([0-9]+)$')

    def __init__(self, **kwargs):
        return ManifoldTable.__init__(self, table='nonorientable_cusped_view', db_path=database_path, **kwargs)