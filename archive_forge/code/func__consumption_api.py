import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def _consumption_api(if_more_than_read: bool=False, datasource_metadata: Optional[str]=None, extra_condition: Optional[str]=None, delegate: Optional[str]=None, pattern='Examples:', insert_after=False):
    """Annotate the function with an indication that it's a consumption API, and that it
    will trigger Dataset execution.
    """
    base = ' will trigger execution of the lazy transformations performed on this dataset.'
    if delegate:
        message = delegate + base
    elif not if_more_than_read:
        message = 'This operation' + base
    else:
        condition = 'If this dataset consists of more than a read, '
        if datasource_metadata is not None:
            condition += f"or if the {datasource_metadata} can't be determined from the metadata provided by the datasource, "
        if extra_condition is not None:
            condition += extra_condition + ', '
        message = condition + 'then this operation' + base

    def wrap(obj):
        _insert_doc_at_pattern(obj, message=message, pattern=pattern, insert_after=insert_after, directive='note')
        return obj
    return wrap