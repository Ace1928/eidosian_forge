from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def _set_app_properties():
    """
    Set properties in driver that propagate to worker processes, allowing for workers to have access to those properties.
    This allows worker integration to have access to app_name and application_id.
    """
    from pyspark import SparkContext
    spark_context = SparkContext._active_spark_context
    if spark_context:
        spark_context.setLocalProperty('sentry_app_name', spark_context.appName)
        spark_context.setLocalProperty('sentry_application_id', spark_context.applicationId)