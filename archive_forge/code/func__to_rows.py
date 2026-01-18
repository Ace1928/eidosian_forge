from typing import Any, Iterable, List
import pyspark.sql as ps
import pyspark.sql.functions as psf
from pyspark import RDD
from pyspark.sql import SparkSession
import warnings
from .convert import to_schema, to_spark_schema
from .misc import is_spark_connect
def _to_rows(rdd: Iterable[Any]) -> Iterable[Any]:
    for item in rdd:
        yield item[1]