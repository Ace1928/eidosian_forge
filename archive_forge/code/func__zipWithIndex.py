from typing import Any, Iterable, List
import pyspark.sql as ps
import pyspark.sql.functions as psf
from pyspark import RDD
from pyspark.sql import SparkSession
import warnings
from .convert import to_schema, to_spark_schema
from .misc import is_spark_connect
def _zipWithIndex(rdd: RDD, to_rows: bool=False) -> RDD:
    """
    Modified from
    https://github.com/davies/spark/blob/cebe5bfe263baf3349353f1473f097396821514a/python/pyspark/rdd.py

    """
    starts = [0]
    if rdd.getNumPartitions() > 1:
        nums = rdd.mapPartitions(lambda it: [sum((1 for i in it))]).collect()
        for i in range(len(nums) - 1):
            starts.append(starts[-1] + nums[i])

    def func1(k, it):
        for i, v in enumerate(it, starts[k]):
            yield (i, v)

    def func2(k, it):
        for i, v in enumerate(it, starts[k]):
            yield (list(v) + [i])
    if not to_rows:
        return rdd.mapPartitionsWithIndex(func1)
    else:
        return rdd.mapPartitionsWithIndex(func2)