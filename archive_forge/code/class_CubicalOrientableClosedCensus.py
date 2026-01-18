from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class CubicalOrientableClosedCensus(PlatonicManifoldTable):
    """
        Iterator for the cubical orientable closed hyperbolic manifolds up
        to 10 cubes, i.e., manifolds that admit a tessellation by regular finite
        hyperbolic cubes.

        >>> len(CubicalOrientableClosedCensus)
        69

        """
    _regex = re.compile('ocube\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'cubical_orientable_closed_census', **kwargs)