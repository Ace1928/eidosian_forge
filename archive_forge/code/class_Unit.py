import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
class Unit(BaseGrid):
    """ Vector of unit values (as in R's grid package) """
    _r_constructor = grid_env['unit']