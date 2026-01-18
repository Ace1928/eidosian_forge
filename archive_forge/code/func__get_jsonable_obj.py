import base64
import datetime
import importlib
import json
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from json import JSONEncoder
from typing import Any, Dict, Optional
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.json_format import MessageToJson, ParseDict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
def _get_jsonable_obj(data, pandas_orient='records'):
    """Attempt to make the data json-able via standard library.

    Look for some commonly used types that are not jsonable and convert them into json-able ones.
    Unknown data types are returned as is.

    Args:
        data: Data to be converted, works with pandas and numpy, rest will be returned as is.
        pandas_orient: If `data` is a Pandas DataFrame, it will be converted to a JSON
            dictionary using this Pandas serialization orientation.
    """
    import numpy as np
    import pandas as pd
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient=pandas_orient)
    if isinstance(data, pd.Series):
        return pd.DataFrame(data).to_dict(orient=pandas_orient)
    else:
        return data