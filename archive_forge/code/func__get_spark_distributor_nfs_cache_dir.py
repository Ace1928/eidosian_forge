import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
def _get_spark_distributor_nfs_cache_dir():
    from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
    if (nfs_root_dir := get_nfs_cache_root_dir()) is not None:
        cache_dir = os.path.join(nfs_root_dir, 'mlflow_distributor_cache_dir')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    return None