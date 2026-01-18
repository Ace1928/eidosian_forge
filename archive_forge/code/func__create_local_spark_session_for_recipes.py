import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
def _create_local_spark_session_for_recipes():
    """Create a sparksession to be used within an recipe step run in a subprocess locally."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    try:
        spark_scala_version = _get_spark_scala_version()
    except Exception as e:
        raise RuntimeError('Failed to get spark scala version.') from e
    _prepare_subprocess_environ_for_creating_local_spark_session()
    return SparkSession.builder.master('local[*]').config('spark.jars.packages', f'io.delta:delta-spark_{spark_scala_version}:3.0.0').config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension').config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog').config('spark.sql.execution.arrow.pyspark.enabled', 'true').getOrCreate()