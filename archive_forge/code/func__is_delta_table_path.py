import logging
import os
from typing import Optional
def _is_delta_table_path(path: str) -> bool:
    """Checks if the specified filesystem path is a Delta table.

    Returns:
        True if the specified path is a Delta table. False otherwise.
    """
    if os.path.exists(path) and os.path.isdir(path) and ('_delta_log' in os.listdir(path)):
        return True
    from mlflow.utils.uri import dbfs_hdfs_uri_to_fuse_path
    try:
        dbfs_path = dbfs_hdfs_uri_to_fuse_path(path)
        return os.path.exists(dbfs_path) and '_delta_log' in os.listdir(dbfs_path)
    except Exception:
        return False