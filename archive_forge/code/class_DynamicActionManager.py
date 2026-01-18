from oslo_utils import uuidutils
from mistralclient.api import base
class DynamicActionManager(base.ResourceManager):
    resource_class = DynamicAction

    def get(self, identifier, namespace=''):
        self._ensure_not_empty(identifier=identifier)
        return self._get('/dynamic_actions/%s?namespace=%s' % (identifier, namespace))

    def create(self, name, class_name, code_source, scope='private', namespace=''):
        self._ensure_not_empty(name=name, class_name=class_name, code_source=code_source)
        data = {'name': name, 'class_name': class_name, 'scope': scope, 'namespace': namespace}
        if uuidutils.is_uuid_like(code_source):
            data['code_source_id'] = code_source
        else:
            data['code_source_name'] = code_source
        return self._create('/dynamic_actions', data)

    def update(self, identifier, class_name=None, code_source=None, scope='private', namespace=''):
        self._ensure_not_empty(identifier=identifier)
        data = {'scope': scope, 'namespace': namespace}
        if uuidutils.is_uuid_like(identifier):
            data['id'] = identifier
        else:
            data['name'] = identifier
        if class_name:
            data['class_name'] = class_name
        if code_source:
            if uuidutils.is_uuid_like(code_source):
                data['code_source_id'] = code_source
            else:
                data['code_source_name'] = code_source
        return self._update('/dynamic_actions', data)

    def list(self, marker='', limit=None, sort_keys='', sort_dirs='', fields='', namespace='', **filters):
        if namespace:
            filters['namespace'] = namespace
        query_string = self._build_query_params(marker=marker, limit=limit, sort_keys=sort_keys, sort_dirs=sort_dirs, fields=fields, filters=filters)
        return self._list('/dynamic_actions%s' % query_string, response_key='dynamic_actions')

    def delete(self, identifier, namespace=''):
        self._ensure_not_empty(identifier=identifier)
        self._delete('/dynamic_actions/%s?namespace=%s' % (identifier, namespace))