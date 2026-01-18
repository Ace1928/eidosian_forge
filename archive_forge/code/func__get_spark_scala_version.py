import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
def _get_spark_scala_version():
    from mlflow.utils.databricks_utils import is_in_databricks_runtime
    if is_in_databricks_runtime() and 'SPARK_SCALA_VERSION' in os.environ:
        return os.environ['SPARK_SCALA_VERSION']
    if (spark := _get_active_spark_session()):
        return _get_spark_scala_version_from_spark_session(spark)
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_get_spark_scala_version_child_proc_target, args=(result_queue,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError('Failed to read scala version.')
    return result_queue.get()