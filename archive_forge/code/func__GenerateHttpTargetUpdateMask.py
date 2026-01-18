from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
import six
def _GenerateHttpTargetUpdateMask(messages, queue, updated_fields, http_uri_override=None, http_method_override=None, http_header_override=None, http_oauth_email_override=None, http_oauth_scope_override=None, http_oidc_email_override=None, http_oidc_audience_override=None):
    """A helper function to generate update mask given the override config."""
    if _HttpTargetNeedsUpdate(updated_fields):
        http_target = messages.HttpTarget()
        if 'httpTarget.uriOverride' in updated_fields:
            http_target.uriOverride = http_uri_override
        if 'httpTarget.httpMethod' in updated_fields:
            http_target.httpMethod = http_method_override
        if 'httpTarget.headerOverrides' in updated_fields:
            if http_header_override is None:
                http_target.headerOverrides = []
            else:
                headers_list = []
                for ho in http_header_override:
                    header_override = messages.HeaderOverride(header=messages.Header(key=ho.header.key, value=ho.header.value))
                    headers_list.append(header_override)
                http_target.headerOverrides = headers_list
        if 'httpTarget.oauthToken.serviceAccountEmail' in updated_fields or 'httpTarget.oauthToken.scope' in updated_fields:
            if 'httpTarget.oauthToken.serviceAccountEmail' not in updated_fields or (http_oauth_email_override is None and http_oauth_scope_override is not None):
                raise RequiredFieldsMissingError('Oauth service account email (http-oauth-service-account-email-override) must be set.')
            elif http_oauth_email_override is None and http_oauth_scope_override is None:
                http_target.oauthToken = None
            else:
                http_target.oauthToken = messages.OAuthToken(serviceAccountEmail=http_oauth_email_override, scope=http_oauth_scope_override)
        if 'httpTarget.oidcToken.serviceAccountEmail' in updated_fields or 'httpTarget.oidcToken.audience' in updated_fields:
            if 'httpTarget.oidcToken.serviceAccountEmail' not in updated_fields or (http_oidc_email_override is None and http_oidc_audience_override is not None):
                raise RequiredFieldsMissingError('Oidc service account email (http-oidc-service-account-email-override) must be set.')
            if http_oidc_email_override is None and http_oidc_audience_override is None:
                http_target.oidcToken = None
            else:
                http_target.oidcToken = messages.OidcToken(serviceAccountEmail=http_oidc_email_override, audience=http_oidc_audience_override)
        queue.httpTarget = None if _IsEmptyConfig(http_target) else http_target