from sys import version_info as _swig_python_version_info
import weakref
def RefuteWrapper(self, solver):
    try:
        self.Refute(solver)
    except Exception as e:
        if 'CP Solver fail' in str(e):
            solver.ShouldFail()
        else:
            raise