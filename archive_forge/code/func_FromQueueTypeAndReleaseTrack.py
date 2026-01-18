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