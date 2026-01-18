from __future__ import absolute_import
import sys
from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
class SparkWorkerIntegration(Integration):
    identifier = 'spark_worker'

    @staticmethod
    def setup_once():
        import pyspark.daemon as original_daemon
        original_daemon.worker_main = _sentry_worker_main