from mistralclient.api import base
class ActionManager(base.ResourceManager):
    resource_class = Action

    def create(self, definition, scope='private', namespace=''):
        self._ensure_not_empty(definition=definition)
        definition = self.get_contents_if_file(definition)
        url = '/actions?scope=%s' % scope
        if namespace:
            url += '&namespace=%s' % namespace
        return self._create(url, definition, response_key='actions', dump_json=False, headers={'content-type': 'text/plain'}, is_iter_resp=True)

    def update(self, definition, scope='private', id=None, namespace=''):
        self._ensure_not_empty(definition=definition)
        params = '?scope=%s' % scope
        if namespace:
            params += '&namespace=%s' % namespace
        url = ('/actions/%s' % id if id else '/actions') + params
        definition = self.get_contents_if_file(definition)
        return self._update(url, definition, response_key='actions', dump_json=False, headers={'content-type': 'text/plain'}, is_iter_resp=True)

    def list(self, marker='', limit=None, sort_keys='', sort_dirs='', fields='', namespace='', **filters):
        if namespace:
            filters['namespace'] = namespace
        query_string = self._build_query_params(marker=marker, limit=limit, sort_keys=sort_keys, sort_dirs=sort_dirs, fields=fields, filters=filters)
        return self._list('/actions%s' % query_string, response_key='actions')

    def get(self, identifier, namespace=''):
        self._ensure_not_empty(identifier=identifier)
        return self._get('/actions/%s/%s' % (identifier, namespace))

    def delete(self, identifier, namespace=''):
        self._ensure_not_empty(identifier=identifier)
        self._delete('/actions/%s/%s' % (identifier, namespace))

    def validate(self, definition):
        self._ensure_not_empty(definition=definition)
        definition = self.get_contents_if_file(definition)
        return self._validate('/actions/validate', definition, dump_json=False, headers={'content-type': 'text/plain'})