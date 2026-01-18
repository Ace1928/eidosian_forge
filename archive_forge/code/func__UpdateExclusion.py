from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
@classmethod
def _UpdateExclusion(cls, compute_client, existing_exclusion, request_headers=None, request_cookies=None, request_query_params=None, request_uris=None):
    """Updates Exclusion."""
    new_exclusion = compute_client.messages.SecurityPolicyRulePreconfiguredWafConfigExclusion()
    new_exclusion.targetRuleSet = existing_exclusion.targetRuleSet
    for target_rule_id in existing_exclusion.targetRuleIds or []:
        new_exclusion.targetRuleIds.append(target_rule_id)
    request_headers_to_remove = []
    for request_header in request_headers or []:
        request_headers_to_remove.append(cls._ConvertRequestFieldToAdd(compute_client, request_header))
    new_exclusion.requestHeadersToExclude.extend(cls._RemoveRequestFields(existing_exclusion.requestHeadersToExclude, request_headers_to_remove))
    request_cookies_to_remove = []
    for request_cookie in request_cookies or []:
        request_cookies_to_remove.append(cls._ConvertRequestFieldToAdd(compute_client, request_cookie))
    new_exclusion.requestCookiesToExclude.extend(cls._RemoveRequestFields(existing_exclusion.requestCookiesToExclude, request_cookies_to_remove))
    request_query_params_to_remove = []
    for request_query_param in request_query_params or []:
        request_query_params_to_remove.append(cls._ConvertRequestFieldToAdd(compute_client, request_query_param))
    new_exclusion.requestQueryParamsToExclude.extend(cls._RemoveRequestFields(existing_exclusion.requestQueryParamsToExclude, request_query_params_to_remove))
    request_uris_to_remove = []
    for request_uri in request_uris or []:
        request_uris_to_remove.append(cls._ConvertRequestFieldToAdd(compute_client, request_uri))
    new_exclusion.requestUrisToExclude.extend(cls._RemoveRequestFields(existing_exclusion.requestUrisToExclude, request_uris_to_remove))
    if not (new_exclusion.requestHeadersToExclude or new_exclusion.requestCookiesToExclude or new_exclusion.requestQueryParamsToExclude or new_exclusion.requestUrisToExclude):
        return None
    return new_exclusion