from sys import version_info as _swig_python_version_info
import weakref
def GetDimensionResourceGroupIndices(self, dimension):
    """
        Returns the indices of resource groups for this dimension. This method can
        only be called after the model has been closed.
        """
    return _pywrapcp.RoutingModel_GetDimensionResourceGroupIndices(self, dimension)