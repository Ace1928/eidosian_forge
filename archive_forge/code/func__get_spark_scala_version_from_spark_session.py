import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
def _get_spark_scala_version_from_spark_session(spark):
    version = spark._jvm.scala.util.Properties.versionNumberString().split('.', 2)
    return f'{version[0]}.{version[1]}'