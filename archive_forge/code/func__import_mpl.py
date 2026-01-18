from statsmodels.compat.python import lrange
def _import_mpl():
    """This function is not needed outside this utils module."""
    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError('Matplotlib is not found.')
    return plt