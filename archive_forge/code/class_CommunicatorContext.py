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
class CommunicatorContext:
    """A context controlling collective communicator initialization and finalization.
    This isn't specificially necessary (note Part 3), but it is more understandable
    coding-wise.

    """

    def __init__(self, context: BarrierTaskContext, **args: Any) -> None:
        self.args = args
        self.args['DMLC_TASK_ID'] = str(context.partitionId())

    def __enter__(self) -> None:
        collective.init(**self.args)

    def __exit__(self, *args: Any) -> None:
        collective.finalize()