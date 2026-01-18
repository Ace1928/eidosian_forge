import numpy as np
from ...metaarray import MetaArray
def adaptiveDetrend(data, x=None, threshold=3.0):
    """Return the signal with baseline removed. Discards outliers from baseline measurement."""
    try:
        import scipy.signal
    except ImportError:
        raise Exception('adaptiveDetrend() requires the package scipy.signal.')
    if x is None:
        x = data.xvals(0)
    d = data.view(np.ndarray)
    d2 = scipy.signal.detrend(d)
    stdev = d2.std()
    mask = abs(d2) < stdev * threshold
    lr = scipy.stats.linregress(x[mask], d[mask])
    base = lr[1] + lr[0] * x
    d4 = d - base
    if hasattr(data, 'implements') and data.implements('MetaArray'):
        return MetaArray(d4, info=data.infoCopy())
    return d4