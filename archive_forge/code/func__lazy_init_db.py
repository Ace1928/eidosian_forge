import numpy as np
def _lazy_init_db():
    global _ufunc_db
    if _ufunc_db is None:
        _ufunc_db = {}
        _fill_ufunc_db(_ufunc_db)