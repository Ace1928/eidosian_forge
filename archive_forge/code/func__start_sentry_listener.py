from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def _start_sentry_listener(sc):
    """
    Start java gateway server to add custom `SparkListener`
    """
    from pyspark.java_gateway import ensure_callback_server_started
    gw = sc._gateway
    ensure_callback_server_started(gw)
    listener = SentryListener()
    sc._jsc.sc().addSparkListener(listener)