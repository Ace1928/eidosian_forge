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
def _get_or_create_tmp_dir() -> str:
    root_dir = SparkFiles.getRootDirectory()
    xgb_tmp_dir = os.path.join(root_dir, 'xgboost-tmp')
    if not os.path.exists(xgb_tmp_dir):
        os.makedirs(xgb_tmp_dir)
    return xgb_tmp_dir