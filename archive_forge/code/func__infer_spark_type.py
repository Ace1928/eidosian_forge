import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
def _infer_spark_type(x, data=None, col_name=None) -> DataType:
    import pyspark.sql.types
    from pyspark.sql.functions import col, collect_list
    if isinstance(x, pyspark.sql.types.NumericType):
        if isinstance(x, pyspark.sql.types.IntegralType):
            if isinstance(x, pyspark.sql.types.LongType):
                return DataType.long
            else:
                return DataType.integer
        elif isinstance(x, pyspark.sql.types.FloatType):
            return DataType.float
        elif isinstance(x, pyspark.sql.types.DoubleType):
            return DataType.double
    elif isinstance(x, pyspark.sql.types.BooleanType):
        return DataType.boolean
    elif isinstance(x, pyspark.sql.types.StringType):
        return DataType.string
    elif isinstance(x, pyspark.sql.types.BinaryType):
        return DataType.binary
    elif isinstance(x, (pyspark.sql.types.DateType, pyspark.sql.types.TimestampType)):
        return DataType.datetime
    elif isinstance(x, pyspark.sql.types.ArrayType):
        return Array(_infer_spark_type(x.elementType))
    elif isinstance(x, pyspark.sql.types.StructType):
        return Object(properties=[Property(name=f.name, dtype=_infer_spark_type(f.dataType), required=not f.nullable) for f in x.fields])
    elif isinstance(x, pyspark.sql.types.MapType):
        if data is None or col_name is None:
            raise MlflowException('Cannot infer schema for MapType without data and column name.')
        if isinstance(x.valueType, pyspark.sql.types.MapType):
            raise MlflowException('Please construct spark DataFrame with schema using StructType for dictionary/map fields, MLflow schema inference only supports scalar, array and struct types.')
        merged_keys = data.selectExpr(f'map_keys({col_name}) as keys').agg(collect_list(col('keys')).alias('merged_keys')).head().merged_keys
        keys = {key for sublist in merged_keys for key in sublist}
        return Object(properties=[Property(name=k, dtype=_infer_spark_type(x.valueType)) for k in keys])
    else:
        raise MlflowException.invalid_parameter_value(f"Unsupported Spark Type '{type(x)}' for MLflow schema.")