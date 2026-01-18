import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
class SparkTaskContextConfigProvider(DatabricksConfigProvider):
    """Loads credentials from Spark TaskContext if running in a Spark Executor."""

    @staticmethod
    def _get_spark_task_context_or_none():
        try:
            from pyspark import TaskContext
            return TaskContext.get()
        except ImportError:
            return None

    @staticmethod
    def set_insecure(x):
        from pyspark import SparkContext
        new_val = 'True' if x else None
        SparkContext._active_spark_context.setLocalProperty('spark.databricks.ignoreTls', new_val)

    def get_config(self):
        context = self._get_spark_task_context_or_none()
        if context is not None:
            host = context.getLocalProperty('spark.databricks.api.url')
            token = context.getLocalProperty('spark.databricks.token')
            insecure = context.getLocalProperty('spark.databricks.ignoreTls')
            config = DatabricksConfig.from_token(host=host, token=token, refresh_token=None, insecure=insecure, jobs_api_version=None)
            if config.is_valid:
                return config
        return None