from sys import version_info as _swig_python_version_info
import weakref
def HasMandatoryDisjunctions(self):
    """
        Returns true if the model contains mandatory disjunctions (ones with
        kNoPenalty as penalty).
        """
    return _pywrapcp.RoutingModel_HasMandatoryDisjunctions(self)