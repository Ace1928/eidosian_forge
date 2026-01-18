from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class LinkExteriorsTable(ManifoldTable):
    """
        Link exteriors usually know a DT code describing the assocated link.
        """
    _select = 'select name, triangulation, DT from %s '

    def _finalize(self, M, row):
        M.set_name(row[0])
        M._set_DTcode(row[2])