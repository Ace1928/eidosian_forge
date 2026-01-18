from collections import namedtuple
import textwrap
def _default_use_pygeos():
    import geopandas._compat as compat
    return compat.USE_PYGEOS