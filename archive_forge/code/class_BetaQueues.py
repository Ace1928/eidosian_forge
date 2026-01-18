from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
import six
class BetaQueues(BaseQueues):
    """Client for queues service in the Cloud Tasks API."""

    def Create(self, parent_ref, queue_ref, retry_config=None, rate_limits=None, app_engine_http_queue=None, stackdriver_logging_config=None, queue_type=None, http_target=None):
        """Prepares and sends a Create request for creating a queue."""
        is_app_engine_target_set = app_engine_http_queue is not None and app_engine_http_queue.appEngineRoutingOverride is not None
        is_http_target_set = http_target is not None
        if is_app_engine_target_set and is_http_target_set:
            raise CreatingHttpAndAppEngineQueueError('Attempting to send multiple queue target types simultaneously: {} , {}'.format(six.text_type(app_engine_http_queue), six.text_type(http_target)))
        if is_http_target_set:
            queue = self.messages.Queue(name=queue_ref.RelativeName(), retryConfig=retry_config, rateLimits=rate_limits, stackdriverLoggingConfig=stackdriver_logging_config, type=queue_type, httpTarget=http_target)
        else:
            queue = self.messages.Queue(name=queue_ref.RelativeName(), retryConfig=retry_config, rateLimits=rate_limits, appEngineHttpQueue=app_engine_http_queue, stackdriverLoggingConfig=stackdriver_logging_config, type=queue_type)
        request = self.messages.CloudtasksProjectsLocationsQueuesCreateRequest(parent=parent_ref.RelativeName(), queue=queue)
        return self.queues_service.Create(request)

    def Patch(self, queue_ref, updated_fields, retry_config=None, rate_limits=None, app_engine_routing_override=None, task_ttl=None, task_tombstone_ttl=None, stackdriver_logging_config=None, queue_type=None, http_uri_override=None, http_method_override=None, http_header_override=None, http_oauth_email_override=None, http_oauth_scope_override=None, http_oidc_email_override=None, http_oidc_audience_override=None):
        """Prepares and sends a Patch request for modifying a queue."""
        if queue_type and queue_type != queue_type.PULL:
            queue_type = None
        if not any([retry_config, rate_limits, task_ttl, task_tombstone_ttl, stackdriver_logging_config]):
            if _NeitherUpdateNorClear([app_engine_routing_override], ['appEngineRoutingOverride'], updated_fields) and _NeitherUpdateNorClear([http_uri_override, http_method_override, http_header_override, http_oauth_email_override, http_oauth_scope_override, http_oidc_email_override, http_oidc_audience_override], http_target_update_masks_list, updated_fields):
                raise NoFieldsSpecifiedError('Must specify at least one field to update.')
        queue = self.messages.Queue(name=queue_ref.RelativeName(), type=queue_type)
        if retry_config is not None:
            queue.retryConfig = retry_config
        if rate_limits is not None:
            queue.rateLimits = rate_limits
        if task_ttl is not None:
            queue.taskTtl = task_ttl
        if task_tombstone_ttl is not None:
            queue.tombstoneTtl = task_tombstone_ttl
        if stackdriver_logging_config is not None:
            queue.stackdriverLoggingConfig = stackdriver_logging_config
        if app_engine_routing_override is not None:
            if _IsEmptyConfig(app_engine_routing_override):
                queue.appEngineHttpQueue = self.messages.AppEngineHttpQueue()
            else:
                queue.appEngineHttpQueue = self.messages.AppEngineHttpQueue(appEngineRoutingOverride=app_engine_routing_override)
        _GenerateHttpTargetUpdateMask(self.messages, queue, updated_fields, http_uri_override, http_method_override, http_header_override, http_oauth_email_override, http_oauth_scope_override, http_oidc_email_override, http_oidc_audience_override)
        update_mask = ','.join(updated_fields)
        request = self.messages.CloudtasksProjectsLocationsQueuesPatchRequest(name=queue_ref.RelativeName(), queue=queue, updateMask=update_mask)
        return self.queues_service.Patch(request)