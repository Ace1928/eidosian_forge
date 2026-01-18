from sys import version_info as _swig_python_version_info
import weakref
class SolutionCollector(SearchMonitor):
    """
    This class is the root class of all solution collectors.
    It implements a basic query API to be used independently
    of the collector used.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')
    __repr__ = _swig_repr

    def DebugString(self):
        return _pywrapcp.SolutionCollector_DebugString(self)

    def Add(self, *args):
        return _pywrapcp.SolutionCollector_Add(self, *args)

    def AddObjective(self, objective):
        return _pywrapcp.SolutionCollector_AddObjective(self, objective)

    def EnterSearch(self):
        """ Beginning of the search."""
        return _pywrapcp.SolutionCollector_EnterSearch(self)

    def SolutionCount(self):
        """ Returns how many solutions were stored during the search."""
        return _pywrapcp.SolutionCollector_SolutionCount(self)

    def Solution(self, n):
        """ Returns the nth solution."""
        return _pywrapcp.SolutionCollector_Solution(self, n)

    def WallTime(self, n):
        """ Returns the wall time in ms for the nth solution."""
        return _pywrapcp.SolutionCollector_WallTime(self, n)

    def Branches(self, n):
        """ Returns the number of branches when the nth solution was found."""
        return _pywrapcp.SolutionCollector_Branches(self, n)

    def Failures(self, n):
        """
        Returns the number of failures encountered at the time of the nth
        solution.
        """
        return _pywrapcp.SolutionCollector_Failures(self, n)

    def ObjectiveValue(self, n):
        """ Returns the objective value of the nth solution."""
        return _pywrapcp.SolutionCollector_ObjectiveValue(self, n)

    def Value(self, n, var):
        """ This is a shortcut to get the Value of 'var' in the nth solution."""
        return _pywrapcp.SolutionCollector_Value(self, n, var)

    def StartValue(self, n, var):
        """ This is a shortcut to get the StartValue of 'var' in the nth solution."""
        return _pywrapcp.SolutionCollector_StartValue(self, n, var)

    def EndValue(self, n, var):
        """ This is a shortcut to get the EndValue of 'var' in the nth solution."""
        return _pywrapcp.SolutionCollector_EndValue(self, n, var)

    def DurationValue(self, n, var):
        """ This is a shortcut to get the DurationValue of 'var' in the nth solution."""
        return _pywrapcp.SolutionCollector_DurationValue(self, n, var)

    def PerformedValue(self, n, var):
        """ This is a shortcut to get the PerformedValue of 'var' in the nth solution."""
        return _pywrapcp.SolutionCollector_PerformedValue(self, n, var)

    def ForwardSequence(self, n, var):
        """
        This is a shortcut to get the ForwardSequence of 'var' in the
        nth solution. The forward sequence is the list of ranked interval
        variables starting from the start of the sequence.
        """
        return _pywrapcp.SolutionCollector_ForwardSequence(self, n, var)

    def BackwardSequence(self, n, var):
        """
        This is a shortcut to get the BackwardSequence of 'var' in the
        nth solution. The backward sequence is the list of ranked interval
        variables starting from the end of the sequence.
        """
        return _pywrapcp.SolutionCollector_BackwardSequence(self, n, var)

    def Unperformed(self, n, var):
        """
        This is a shortcut to get the list of unperformed of 'var' in the
        nth solution.
        """
        return _pywrapcp.SolutionCollector_Unperformed(self, n, var)