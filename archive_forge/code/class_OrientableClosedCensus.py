from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class OrientableClosedCensus(ClosedManifoldTable):
    """
        Iterator for 11,031 closed hyperbolic manifolds from the census by
        Hodgson and Weeks.

        >>> len(OrientableClosedCensus)
        11031
        >>> len(OrientableClosedCensus(betti=2))
        1
        >>> for M in OrientableClosedCensus(betti=2):
        ...   print(M, M.homology())
        ... 
        v1539(5,1) Z + Z
        """

    def __init__(self, **kwargs):
        return ClosedManifoldTable.__init__(self, table='orientable_closed_view', db_path=database_path, **kwargs)