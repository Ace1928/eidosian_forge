import concurrent.futures
import logging
import sys
import threading
import uuid
from py4j.java_gateway import CallbackServerParameters
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.spark import FLAVOR_NAME
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import _truncate_and_ellipsize
from mlflow.utils.autologging_utils import (
from mlflow.utils.databricks_utils import get_repl_id as get_databricks_repl_id
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH
class PythonSubscriber(metaclass=ExceptionSafeClass):
    """
    Subscriber, intended to be instantiated once per Python process, that logs Spark table
    information propagated from Java to the current MLflow run, starting a run if necessary.
    class implements a Java interface (org.mlflow.spark.autologging.MlflowAutologEventSubscriber,
    defined in the mlflow-spark package) that's called-into by autologging logic in the JVM in order
    to propagate Spark datasource read events to Python.

    This class leverages the Py4j callback mechanism to receive callbacks from the JVM, see
    https://www.py4j.org/advanced_topics.html#implementing-java-interfaces-from-python-callback for
    more information.
    """

    def __init__(self):
        self._repl_id = _get_repl_id()

    def toString(self):
        return f'PythonSubscriber<replId={self.replId()}>'

    def ping(self):
        return None

    def notify(self, path, version, data_format):
        try:
            self._notify(path, version, data_format)
        except Exception as e:
            _logger.error('Unexpected exception %s while attempting to log Spark datasource info. Exception:\n', e)

    def _notify(self, path, version, data_format):
        """
        Method called by Scala SparkListener to propagate datasource read events to the current
        Python process
        """
        if autologging_is_disabled(FLAVOR_NAME):
            return
        active_run = mlflow.active_run()
        if active_run:
            _set_run_tag_async(active_run.info.run_id, path, version, data_format)
        else:
            add_table_info_to_context_provider(path, version, data_format)

    def replId(self):
        return self._repl_id

    class Java:
        implements = [f'{_JAVA_PACKAGE}.MlflowAutologEventSubscriber']