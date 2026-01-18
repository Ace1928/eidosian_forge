import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
def _create_local_spark_session_for_loading_spark_model():
    from pyspark.sql import SparkSession
    return SparkSession.builder.config('spark.python.worker.reuse', 'true').config('spark.databricks.io.cache.enabled', 'false').config('spark.executor.allowSparkContext', 'true').config('spark.driver.host', '127.0.0.1').config('spark.executor.allowSparkContext', 'true').config('spark.driver.extraJavaOptions', '-Dlog4j.configuration=file:/usr/local/spark/conf/log4j.properties').master('local[1]').getOrCreate()