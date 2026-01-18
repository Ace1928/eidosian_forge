from sys import version_info as _swig_python_version_info
import weakref
class PyConstraint(Constraint):

    def __init__(self, solver):
        super().__init__(solver)
        self.__demons = []

    def Demon(self, method, *args):
        demon = PyConstraintDemon(self, method, False, *args)
        self.__demons.append(demon)
        return demon

    def DelayedDemon(self, method, *args):
        demon = PyConstraintDemon(self, method, True, *args)
        self.__demons.append(demon)
        return demon

    def InitialPropagateDemon(self):
        return self.solver().ConstraintInitialPropagateCallback(self)

    def DelayedInitialPropagateDemon(self):
        return self.solver().DelayedConstraintInitialPropagateCallback(self)

    def InitialPropagateWrapper(self):
        try:
            self.InitialPropagate()
        except Exception as e:
            if 'CP Solver fail' in str(e):
                self.solver().ShouldFail()
            else:
                raise

    def DebugString(self):
        return 'PyConstraint'