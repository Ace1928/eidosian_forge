import logging
import os
from typing import Optional
def _get_delta_table_latest_version(j_delta_table) -> int:
    """Obtains the latest version of the specified Delta table Java class.

    Args:
        delta_table: A Java DeltaTable class instance.

    Returns:
        The version of the Delta table.

    """
    latest_commit_jdf = j_delta_table.history(1)
    latest_commit_row = latest_commit_jdf.head()
    version_field_idx = latest_commit_row.fieldIndex('version')
    return latest_commit_row.get(version_field_idx)