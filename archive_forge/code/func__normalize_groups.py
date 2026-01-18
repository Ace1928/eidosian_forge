import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _normalize_groups(self, identity_value):
    if 'name' in identity_value['groups']:
        group_names_list = self._ast_literal_eval(identity_value['groups'])

        def convert_json(group):
            if group.startswith('JSON:'):
                return jsonutils.loads(group.lstrip('JSON:'))
            return group
        group_dicts = [convert_json(g) for g in group_names_list]
        for g in group_dicts:
            if 'domain' not in g:
                msg = _("Invalid rule: %(identity_value)s. Both 'groups' and 'domain' keywords must be specified.")
                msg = msg % {'identity_value': identity_value}
                raise exception.ValidationError(msg)
    else:
        if 'domain' not in identity_value:
            msg = _("Invalid rule: %(identity_value)s. Both 'groups' and 'domain' keywords must be specified.")
            msg = msg % {'identity_value': identity_value}
            raise exception.ValidationError(msg)
        group_names_list = self._ast_literal_eval(identity_value['groups'])
        domain = identity_value['domain']
        group_dicts = [{'name': name, 'domain': domain} for name in group_names_list]
    return group_dicts