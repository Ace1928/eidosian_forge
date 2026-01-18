import flask
import flask_restful
import http.client
from oslo_serialization import jsonutils
from oslo_log import log
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import schema
from keystone.federation import utils
from keystone.server import flask as ks_flask
def _internal_normalize_and_validate_attribute_mapping(self, action_executed_message='created'):
    mapping = self.request_body_json.get('mapping', {})
    mapping = self._normalize_dict(mapping)
    if not mapping.get('schema_version'):
        default_schema_version = utils.get_default_attribute_mapping_schema_version()
        LOG.debug("A mapping [%s] was %s without providing a 'schema_version'; therefore, we need to set one. The current default is [%s]. We will use this value for the attribute mapping being registered. It is recommended that one does not rely on this default value, as it can change, and the already persisted attribute mappings will remain with the previous default values.", mapping, action_executed_message, default_schema_version)
        mapping['schema_version'] = default_schema_version
    utils.validate_mapping_structure(mapping)
    return mapping