import scipy.stats as stats
from ..exceptions import PlotnineError
def _hasattrs(obj, attrs):
    return all((hasattr(obj, attr) for attr in attrs))