from sys import version_info as _swig_python_version_info
import weakref
def RunWrapper(self, solver):
    try:
        self.Run(solver)
    except Exception as e:
        if 'CP Solver fail' in str(e):
            solver.ShouldFail()
        else:
            raise