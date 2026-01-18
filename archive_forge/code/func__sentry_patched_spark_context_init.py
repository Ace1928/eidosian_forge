from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def _sentry_patched_spark_context_init(self, *args, **kwargs):
    init = spark_context_init(self, *args, **kwargs)
    if Hub.current.get_integration(SparkIntegration) is None:
        return init
    _start_sentry_listener(self)
    _set_app_properties()
    with configure_scope() as scope:

        @scope.add_event_processor
        def process_event(event, hint):
            with capture_internal_exceptions():
                if Hub.current.get_integration(SparkIntegration) is None:
                    return event
                event.setdefault('user', {}).setdefault('id', self.sparkUser())
                event.setdefault('tags', {}).setdefault('executor.id', self._conf.get('spark.executor.id'))
                event['tags'].setdefault('spark-submit.deployMode', self._conf.get('spark.submit.deployMode'))
                event['tags'].setdefault('driver.host', self._conf.get('spark.driver.host'))
                event['tags'].setdefault('driver.port', self._conf.get('spark.driver.port'))
                event['tags'].setdefault('spark_version', self.version)
                event['tags'].setdefault('app_name', self.appName)
                event['tags'].setdefault('application_id', self.applicationId)
                event['tags'].setdefault('master', self.master)
                event['tags'].setdefault('spark_home', self.sparkHome)
                event.setdefault('extra', {}).setdefault('web_url', self.uiWebUrl)
            return event
    return init