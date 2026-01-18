import re
from functools import reduce
from typing import Set, Union
from pyspark.ml.base import Transformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import types as t

    Finds feature columns from an input dataset. If a dataset
    contains non-feature columns, those columns are not returned, but
    if `input_fields` is set to include non-feature columns those
    will be included in the return set of column names.

    Args:
        df: An input spark dataframe.
        transformer: A pipeline/transformer to get the required feature columns

    Returns:
        A set of all the feature columns that are required
        for the pipeline/transformer plus any initial columns passed in.
    