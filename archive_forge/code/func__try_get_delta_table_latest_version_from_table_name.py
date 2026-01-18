import logging
import os
from typing import Optional
def _try_get_delta_table_latest_version_from_table_name(table_name: str) -> Optional[int]:
    """Gets the latest version of the Delta table with the specified name.

    Args:
        table_name: The name of the Delta table.

    Returns:
        The version of the Delta table, or None if it cannot be resolved (e.g. because the
        Delta core library is not installed or no such table exists).
    """
    from pyspark.sql import SparkSession
    try:
        spark = SparkSession.builder.getOrCreate()
        j_delta_table = spark._jvm.io.delta.tables.DeltaTable.forName(spark._jsparkSession, table_name)
        return _get_delta_table_latest_version(j_delta_table)
    except Exception as e:
        _logger.warning("Failed to obtain version information for Delta table with name '%s'. Version information may not be included in the dataset source for MLflow Tracking. Exception: %s", table_name, e)