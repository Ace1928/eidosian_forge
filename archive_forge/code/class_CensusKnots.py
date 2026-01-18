from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class CensusKnots(ManifoldTable):
    """
        Iterator for all of the knot exteriors in the SnapPea Census,
        as tabulated by Callahan, Dean, Weeks, Champanerkar, Kofman,
        Patterson, and Dunfield.  These are the knot exteriors which
        can be triangulated by at most 9 ideal tetrahedra.

        >>> for M in CensusKnots[3.4:3.5]: # doctest: +NUMERIC6
        ...   print(M, M.volume(), LinkExteriors.identify(M))
        ... 
        K4_3(0,0) 3.47424776 False
        K5_1(0,0) 3.41791484 False
        K5_2(0,0) 3.42720525 8_1(0,0)
        K5_3(0,0) 3.48666015 9_2(0,0)

        >>> len(CensusKnots)
        1267
        >>> CensusKnots[-1].num_tetrahedra()
        9
        """
    _regex = re.compile('[kK][2-9]_([0-9]+)$')

    def __init__(self, **kwargs):
        return ManifoldTable.__init__(self, table='census_knots_view', db_path=database_path, **kwargs)