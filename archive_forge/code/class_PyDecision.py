from sys import version_info as _swig_python_version_info
import weakref
class PyDecision(Decision):

    def ApplyWrapper(self, solver):
        try:
            self.Apply(solver)
        except Exception as e:
            if 'CP Solver fail' in str(e):
                solver.ShouldFail()
            else:
                raise

    def RefuteWrapper(self, solver):
        try:
            self.Refute(solver)
        except Exception as e:
            if 'CP Solver fail' in str(e):
                solver.ShouldFail()
            else:
                raise

    def DebugString(self):
        return 'PyDecision'