from .. import functions as fn
from .parameterTypes import GroupParameter
from .SystemSolver import SystemSolver

    ParameterSystem is a subclass of GroupParameter that manages a tree of 
    sub-parameters with a set of interdependencies--changing any one parameter
    may affect other parameters in the system.
    
    See parametertree/SystemSolver for more information.
    
    NOTE: This API is experimental and may change substantially across minor 
    version numbers. 
    