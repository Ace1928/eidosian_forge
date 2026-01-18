import contextlib
from multiprocessing import Pool, RLock
from tqdm.auto import tqdm
from ..utils import experimental, logging
def _map_with_joblib(function, iterable, num_proc, types, disable_tqdm, desc, single_map_nested_func):
    import joblib
    with joblib.parallel_backend(ParallelBackendConfig.backend_name, n_jobs=num_proc):
        return joblib.Parallel()((joblib.delayed(single_map_nested_func)((function, obj, types, None, True, None)) for obj in iterable))