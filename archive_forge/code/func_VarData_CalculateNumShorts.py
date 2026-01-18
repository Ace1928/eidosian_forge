from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
def VarData_CalculateNumShorts(self, optimize=True):
    """Deprecated name for VarData_calculateNumShorts() which
    defaults to optimize=True.  Use varData.calculateNumShorts()
    or varData.optimize()."""
    return VarData_calculateNumShorts(self, optimize=optimize)