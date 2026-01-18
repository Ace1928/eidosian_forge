import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
def _get_active_spark_session():
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    try:
        return SparkSession.getActiveSession()
    except Exception:
        return SparkSession._instantiatedSession