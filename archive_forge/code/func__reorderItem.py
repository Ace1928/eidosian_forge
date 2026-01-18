from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
def _reorderItem(lst, mapping):
    return [lst[i] for i in mapping]