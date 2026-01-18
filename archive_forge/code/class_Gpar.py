import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
class Gpar(BaseGrid):
    """ Graphical parameters """
    _r_constructor = grid_env['gpar']
    _get_gpar = grid_env['get.gpar']

    def get(self, names=None):
        return self._get_gpar(names)