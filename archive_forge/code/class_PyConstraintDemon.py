from sys import version_info as _swig_python_version_info
import weakref
class PyConstraintDemon(PyDemon):

    def __init__(self, ct, method, delayed, *args):
        super().__init__()
        self.__constraint = ct
        self.__method = method
        self.__delayed = delayed
        self.__args = args

    def Run(self, solver):
        self.__method(self.__constraint, *self.__args)

    def Priority(self):
        return Solver.DELAYED_PRIORITY if self.__delayed else Solver.NORMAL_PRIORITY

    def DebugString(self):
        return 'PyConstraintDemon'