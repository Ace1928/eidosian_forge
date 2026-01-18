import numpy as np
def _continuous_to_discrete(vals, val_range, n):
    """
    Convert a continuous one-dimensional array to discrete integer values
    based their ranges

    Parameters
    ----------
    vals : Array of continuous values

    val_range : Tuple containing range of continuous values

    n : Number of discrete values

    Returns
    -------
    One-dimensional array of discrete ints

    """
    width = val_range[1] - val_range[0]
    if width == 0:
        return np.zeros_like(vals, dtype=np.uint32)
    res = (vals - val_range[0]) * (n / width)
    np.clip(res, 0, n, out=res)
    return res.astype(np.uint32)