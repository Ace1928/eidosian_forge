from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class DodecahedralOrientableCuspedCensus(PlatonicManifoldTable):
    """
        Iterator for the dodecahedral orientable cusped hyperbolic manifolds up to
        2 dodecahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic dodecahedra.

        Complement of one of the dodecahedral knots by Aitchison and Rubinstein::

          >>> M=DodecahedralOrientableCuspedCensus['odode02_00913']
          >>> M.dehn_fill((1,0))
          >>> M.fundamental_group()
          Generators:
          <BLANKLINE>
          Relators:
          <BLANKLINE>

        """
    _regex = re.compile('odode\\d+_\\d+')

    def __init__(self, **kwargs):
        return PlatonicManifoldTable.__init__(self, 'dodecahedral_orientable_cusped_census', **kwargs)