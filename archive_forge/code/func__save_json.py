import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import pyarrow as pa
import ray.data as rd
from pyarrow import csv as pacsv
from pyarrow import json as pajson
from ray.data.datasource import FileExtensionFilter
from triad.collections import Schema
from triad.collections.dict import ParamDict
from triad.utils.assertion import assert_or_throw
from triad.utils.io import exists, makedirs, rm
from fugue import ExecutionEngine
from fugue._utils.io import FileParser, save_df
from fugue.collections.partition import PartitionSpec
from fugue.dataframe import DataFrame
from fugue_ray.dataframe import RayDataFrame
def _save_json(self, df: RayDataFrame, uri: str, **kwargs: Any) -> None:
    df.native.write_json(uri, ray_remote_args=self._remote_args(), **kwargs)