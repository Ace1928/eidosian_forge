import inspect
import logging
import os
import sys
import uuid
from threading import Thread
from typing import Any, Callable, Dict, Optional, Set, Type
import pyspark
from pyspark import BarrierTaskContext, SparkContext, SparkFiles, TaskContext
from pyspark.sql.session import SparkSession
from xgboost import Booster, XGBModel, collective
from xgboost.tracker import RabitTracker
def _get_default_params_from_func(func: Callable, unsupported_set: Set[str]) -> Dict[str, Any]:
    """Returns a dictionary of parameters and their default value of function fn.  Only
    the parameters with a default value will be included.

    """
    sig = inspect.signature(func)
    filtered_params_dict = {}
    for parameter in sig.parameters.values():
        if parameter.default is not parameter.empty and parameter.name not in unsupported_set:
            filtered_params_dict[parameter.name] = parameter.default
    return filtered_params_dict