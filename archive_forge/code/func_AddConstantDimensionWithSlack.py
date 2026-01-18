from sys import version_info as _swig_python_version_info
import weakref
def AddConstantDimensionWithSlack(self, value, capacity, slack_max, fix_start_cumul_to_zero, name):
    """
        Creates a dimension where the transit variable is constrained to be
        equal to 'value'; 'capacity' is the upper bound of the cumul variables.
        'name' is the name used to reference the dimension; this name is used to
        get cumul and transit variables from the routing model.
        Returns a pair consisting of an index to the registered unary transit
        callback and a bool denoting whether the dimension has been created.
        It is false if a dimension with the same name has already been created
        (and doesn't create the new dimension but still register a new callback).
        """
    return _pywrapcp.RoutingModel_AddConstantDimensionWithSlack(self, value, capacity, slack_max, fix_start_cumul_to_zero, name)