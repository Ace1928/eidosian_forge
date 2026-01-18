from contextlib import contextmanager
from operator import length_hint
from os import cpu_count
from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning
def _executor_map(PoolExecutor, fn, *iterables, **tqdm_kwargs):
    """
    Implementation of `thread_map` and `process_map`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    max_workers  : [default: min(32, cpu_count() + 4)].
    chunksize  : [default: 1].
    lock_name  : [default: "":str].
    """
    kwargs = tqdm_kwargs.copy()
    if 'total' not in kwargs:
        kwargs['total'] = length_hint(iterables[0])
    tqdm_class = kwargs.pop('tqdm_class', tqdm_auto)
    max_workers = kwargs.pop('max_workers', min(32, cpu_count() + 4))
    chunksize = kwargs.pop('chunksize', 1)
    lock_name = kwargs.pop('lock_name', '')
    with ensure_lock(tqdm_class, lock_name=lock_name) as lk:
        with PoolExecutor(max_workers=max_workers, initializer=tqdm_class.set_lock, initargs=(lk,)) as ex:
            return list(tqdm_class(ex.map(fn, *iterables, chunksize=chunksize), **kwargs))