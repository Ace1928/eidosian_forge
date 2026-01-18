from sys import version_info as _swig_python_version_info
import weakref
def SetPrimaryConstrainedDimension(self, dimension_name):
    """
        Set the given dimension as "primary constrained". As of August 2013, this
        is only used by ArcIsMoreConstrainedThanArc().
        "dimension" must be the name of an existing dimension, or be empty, in
        which case there will not be a primary dimension after this call.
        """
    return _pywrapcp.RoutingModel_SetPrimaryConstrainedDimension(self, dimension_name)