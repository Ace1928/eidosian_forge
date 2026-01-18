import flask
from flask import make_response
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import utils
from keystone.conf import CONF
from keystone import exception
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _check_mapped_properties(self, cert_dn, user, user_domain):
    mapping_id = CONF.oauth2.get('oauth2_cert_dn_mapping_id')
    try:
        mapping = PROVIDERS.federation_api.get_mapping(mapping_id)
    except exception.MappingNotFound:
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: mapping id %s is not found. ', mapping_id)
        raise error
    rule_processor = federation_utils.RuleProcessor(mapping.get('id'), mapping.get('rules'))
    try:
        mapped_properties = rule_processor.process(cert_dn)
    except exception.Error as error:
        LOG.exception(error)
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: mapping rule process failed. mapping_id: %s, rules: %s, data: %s.', mapping_id, mapping.get('rules'), jsonutils.dumps(cert_dn))
        raise error
    except Exception as error:
        LOG.exception(error)
        error = exception.OAuth2OtherError(int(http.client.INTERNAL_SERVER_ERROR), http.client.responses[http.client.INTERNAL_SERVER_ERROR], str(error))
        LOG.info('Get OAuth2.0 Access Token API: mapping rule process failed. mapping_id: %s, rules: %s, data: %s.', mapping_id, mapping.get('rules'), jsonutils.dumps(cert_dn))
        raise error
    mapping_user = mapped_properties.get('user', {})
    mapping_user_name = mapping_user.get('name')
    mapping_user_id = mapping_user.get('id')
    mapping_user_email = mapping_user.get('email')
    mapping_domain = mapping_user.get('domain', {})
    mapping_user_domain_id = mapping_domain.get('id')
    mapping_user_domain_name = mapping_domain.get('name')
    if mapping_user_name and mapping_user_name != user.get('name'):
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.', 'user name', mapping_user_name, user.get('name'))
        raise error
    if mapping_user_id and mapping_user_id != user.get('id'):
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.', 'user id', mapping_user_id, user.get('id'))
        raise error
    if mapping_user_email and mapping_user_email != user.get('email'):
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.', 'user email', mapping_user_email, user.get('email'))
        raise error
    if mapping_user_domain_id and mapping_user_domain_id != user_domain.get('id'):
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.', 'user domain id', mapping_user_domain_id, user_domain.get('id'))
        raise error
    if mapping_user_domain_name and mapping_user_domain_name != user_domain.get('name'):
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: %s check failed. DN value: %s, DB value: %s.', 'user domain name', mapping_user_domain_name, user_domain.get('name'))
        raise error