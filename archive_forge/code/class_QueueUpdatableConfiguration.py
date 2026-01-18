from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
class QueueUpdatableConfiguration(object):
    """Data Class for queue configuration updates."""

    @classmethod
    def FromQueueTypeAndReleaseTrack(cls, queue_type, release_track=base.ReleaseTrack.GA):
        """Creates QueueUpdatableConfiguration from the given parameters."""
        config = cls()
        config.retry_config = {}
        config.rate_limits = {}
        config.app_engine_routing_override = {}
        config.http_target = {}
        config.stackdriver_logging_config = {}
        config.retry_config_mask_prefix = None
        config.rate_limits_mask_prefix = None
        config.app_engine_routing_override_mask_prefix = None
        config.http_target_mask_prefix = None
        config.stackdriver_logging_config_mask_prefix = None
        if queue_type == constants.PULL_QUEUE:
            config.retry_config = {'max_attempts': 'maxAttempts', 'max_retry_duration': 'maxRetryDuration'}
            config.retry_config_mask_prefix = 'retryConfig'
        elif queue_type == constants.PUSH_QUEUE:
            if release_track == base.ReleaseTrack.ALPHA:
                config.retry_config = {'max_attempts': 'maxAttempts', 'max_retry_duration': 'maxRetryDuration', 'max_doublings': 'maxDoublings', 'min_backoff': 'minBackoff', 'max_backoff': 'maxBackoff'}
                config.rate_limits = {'max_tasks_dispatched_per_second': 'maxTasksDispatchedPerSecond', 'max_concurrent_tasks': 'maxConcurrentTasks'}
                config.app_engine_routing_override = {'routing_override': 'appEngineRoutingOverride'}
                config.http_target = {'http_uri_override': 'uriOverride', 'http_method_override': 'httpMethod', 'http_header_override': 'headerOverrides', 'http_oauth_service_account_email_override': 'oauthToken.serviceAccountEmail', 'http_oauth_token_scope_override': 'oauthToken.scope', 'http_oidc_service_account_email_override': 'oidcToken.serviceAccountEmail', 'http_oidc_token_audience_override': 'oidcToken.audience'}
                config.retry_config_mask_prefix = 'retryConfig'
                config.rate_limits_mask_prefix = 'rateLimits'
                config.app_engine_routing_override_mask_prefix = 'appEngineHttpTarget'
                config.http_target_mask_prefix = 'httpTarget'
            elif release_track == base.ReleaseTrack.BETA:
                config.retry_config = {'max_attempts': 'maxAttempts', 'max_retry_duration': 'maxRetryDuration', 'max_doublings': 'maxDoublings', 'min_backoff': 'minBackoff', 'max_backoff': 'maxBackoff'}
                config.rate_limits = {'max_dispatches_per_second': 'maxDispatchesPerSecond', 'max_concurrent_dispatches': 'maxConcurrentDispatches', 'max_burst_size': 'maxBurstSize'}
                config.app_engine_routing_override = {'routing_override': 'appEngineRoutingOverride'}
                config.http_target = {'http_uri_override': 'uriOverride', 'http_method_override': 'httpMethod', 'http_header_override': 'headerOverrides', 'http_oauth_service_account_email_override': 'oauthToken.serviceAccountEmail', 'http_oauth_token_scope_override': 'oauthToken.scope', 'http_oidc_service_account_email_override': 'oidcToken.serviceAccountEmail', 'http_oidc_token_audience_override': 'oidcToken.audience'}
                config.stackdriver_logging_config = {'log_sampling_ratio': 'samplingRatio'}
                config.retry_config_mask_prefix = 'retryConfig'
                config.rate_limits_mask_prefix = 'rateLimits'
                config.app_engine_routing_override_mask_prefix = 'appEngineHttpQueue'
                config.http_target_mask_prefix = 'httpTarget'
                config.stackdriver_logging_config_mask_prefix = 'stackdriverLoggingConfig'
            else:
                config.retry_config = {'max_attempts': 'maxAttempts', 'max_retry_duration': 'maxRetryDuration', 'max_doublings': 'maxDoublings', 'min_backoff': 'minBackoff', 'max_backoff': 'maxBackoff'}
                config.rate_limits = {'max_dispatches_per_second': 'maxDispatchesPerSecond', 'max_concurrent_dispatches': 'maxConcurrentDispatches'}
                config.app_engine_routing_override = {'routing_override': 'appEngineRoutingOverride'}
                config.http_target = {'http_uri_override': 'uriOverride', 'http_method_override': 'httpMethod', 'http_header_override': 'headerOverrides', 'http_oauth_service_account_email_override': 'oauthToken.serviceAccountEmail', 'http_oauth_token_scope_override': 'oauthToken.scope', 'http_oidc_service_account_email_override': 'oidcToken.serviceAccountEmail', 'http_oidc_token_audience_override': 'oidcToken.audience'}
                config.stackdriver_logging_config = {'log_sampling_ratio': 'samplingRatio'}
                config.retry_config_mask_prefix = 'retryConfig'
                config.rate_limits_mask_prefix = 'rateLimits'
                config.app_engine_routing_override_mask_prefix = ''
                config.http_target_mask_prefix = 'httpTarget'
                config.stackdriver_logging_config_mask_prefix = 'stackdriverLoggingConfig'
        return config

    def _InitializedConfigsAndPrefixTuples(self):
        """Returns the initialized configs as a list of (config, prefix) tuples."""
        all_configs_and_prefixes = [(self.retry_config, self.retry_config_mask_prefix), (self.rate_limits, self.rate_limits_mask_prefix), (self.app_engine_routing_override, self.app_engine_routing_override_mask_prefix), (self.http_target, self.http_target_mask_prefix), (self.stackdriver_logging_config, self.stackdriver_logging_config_mask_prefix)]
        return [(config, prefix) for config, prefix in all_configs_and_prefixes if config]

    def _GetSingleConfigToMaskMapping(self, config, prefix):
        """Build a map from each arg and its clear_ counterpart to a mask field."""
        fields_to_mask = dict()
        for field in config.keys():
            output_field = config[field]
            if prefix:
                fields_to_mask[field] = '{}.{}'.format(prefix, output_field)
            else:
                fields_to_mask[field] = output_field
            fields_to_mask[_EquivalentClearArg(field)] = fields_to_mask[field]
        return fields_to_mask

    def GetConfigToUpdateMaskMapping(self):
        """Builds mapping from config fields to corresponding update mask fields."""
        config_to_mask = dict()
        for config, prefix in self._InitializedConfigsAndPrefixTuples():
            config_to_mask.update(self._GetSingleConfigToMaskMapping(config, prefix))
        return config_to_mask

    def AllConfigs(self):
        return list(self.retry_config.keys()) + list(self.rate_limits.keys()) + list(self.app_engine_routing_override.keys()) + list(self.http_target.keys()) + list(self.stackdriver_logging_config.keys())