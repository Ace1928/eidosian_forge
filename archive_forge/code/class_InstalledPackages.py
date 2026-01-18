import os
import typing
import warnings
from types import ModuleType
from warnings import warn
import rpy2.rinterface as rinterface
from . import conversion
from rpy2.robjects.functions import (SignatureTranslatedFunction,
from rpy2.robjects import Environment
from rpy2.robjects.packages_utils import (
import rpy2.robjects.help as rhelp
class InstalledPackages(object):
    """ R packages installed. """

    def __init__(self, lib_loc=None):
        libraryiqr = _library(**{'lib.loc': lib_loc})
        lib_results_i = libraryiqr.do_slot('names').index('results')
        self.lib_results = libraryiqr[lib_results_i]
        self.nrows, self.ncols = self.lib_results.do_slot('dim')
        self.colnames = self.lib_results.do_slot('dimnames')[1]
        self.lib_packname_i = self.colnames.index('Package')

    def isinstalled(self, packagename: str):
        if not isinstance(packagename, rinterface.StrSexpVector):
            rinterface.StrSexpVector((packagename,))
        elif len(packagename) > 1:
            raise ValueError('Only specify one package name at a time.')
        nrows = self.nrows
        lib_results, lib_packname_i = (self.lib_results, self.lib_packname_i)
        for i in range(0 + lib_packname_i * nrows, nrows * (lib_packname_i + 1), 1):
            if lib_results[i] == packagename:
                return True
        return False

    def __iter__(self):
        """ Iterate through rows, yield tuples at each iteration """
        lib_results = self.lib_results
        nrows, ncols = (self.nrows, self.ncols)
        colrg = range(0, ncols)
        for row_i in range(nrows):
            yield tuple((lib_results[x * nrows + row_i] for x in colrg))