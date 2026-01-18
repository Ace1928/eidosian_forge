from sys import version_info as _swig_python_version_info
import weakref
def NextWrapper(self, solver):
    try:
        return self.Next(solver)
    except Exception as e:
        if 'CP Solver fail' in str(e):
            return solver.FailDecision()
        else:
            raise