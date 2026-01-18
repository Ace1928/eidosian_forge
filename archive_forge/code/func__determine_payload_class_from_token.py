import os
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.token.providers import base
from keystone.token import token_formatters as tf
def _determine_payload_class_from_token(self, token):
    if token.oauth_scoped:
        return tf.OauthScopedPayload
    elif token.trust_scoped:
        return tf.TrustScopedPayload
    elif token.is_federated:
        if token.project_scoped:
            return tf.FederatedProjectScopedPayload
        elif token.domain_scoped:
            return tf.FederatedDomainScopedPayload
        elif token.unscoped:
            return tf.FederatedUnscopedPayload
    elif token.application_credential_id:
        return tf.ApplicationCredentialScopedPayload
    elif token.oauth2_thumbprint:
        return tf.Oauth2CredentialsScopedPayload
    elif token.project_scoped:
        return tf.ProjectScopedPayload
    elif token.domain_scoped:
        return tf.DomainScopedPayload
    elif token.system_scoped:
        return tf.SystemScopedPayload
    else:
        return tf.UnscopedPayload