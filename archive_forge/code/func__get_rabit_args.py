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
def _get_rabit_args(context: BarrierTaskContext, n_workers: int) -> Dict[str, Any]:
    """Get rabit context arguments to send to each worker."""
    env = _start_tracker(context, n_workers)
    return env