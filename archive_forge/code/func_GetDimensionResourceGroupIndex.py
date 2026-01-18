from sys import version_info as _swig_python_version_info
import weakref
def GetDimensionResourceGroupIndex(self, dimension):
    """
        Returns the index of the resource group attached to the dimension.
        DCHECKS that there's exactly one resource group for this dimension.
        """
    return _pywrapcp.RoutingModel_GetDimensionResourceGroupIndex(self, dimension)