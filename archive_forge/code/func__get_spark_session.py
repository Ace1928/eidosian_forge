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
def _get_spark_session() -> SparkSession:
    """Get or create spark session. Note: This function can only be invoked from driver
    side.

    """
    if pyspark.TaskContext.get() is not None:
        raise RuntimeError('_get_spark_session should not be invoked from executor side.')
    return SparkSession.builder.getOrCreate()