from mistralclient.api import base
class CodeSourceManager(base.ResourceManager):
    resource_class = CodeSource

    def create(self, name, content, namespace='', scope='private'):
        self._ensure_not_empty(name=name, content=content)
        content = self.get_contents_if_file(content)
        return self._create('/code_sources?name=%s&scope=%s&namespace=%s' % (name, scope, namespace), content, dump_json=False, headers={'content-type': 'text/plain'})

    def update(self, identifier, content, namespace='', scope='private'):
        self._ensure_not_empty(identifier=identifier, content=content)
        content = self.get_contents_if_file(content)
        return self._update('/code_sources?identifier=%s&scope=%s&namespace=%s' % (identifier, scope, namespace), content, dump_json=False, headers={'content-type': 'text/plain'})

    def list(self, namespace='', marker='', limit=None, sort_keys='', sort_dirs='', fields='', **filters):
        if namespace:
            filters['namespace'] = namespace
        query_string = self._build_query_params(marker=marker, limit=limit, sort_keys=sort_keys, sort_dirs=sort_dirs, fields=fields, filters=filters)
        return self._list('/code_sources%s' % query_string, response_key='code_sources')

    def get(self, identifier, namespace=''):
        self._ensure_not_empty(identifier=identifier)
        return self._get('/code_sources/%s?namespace=%s' % (identifier, namespace))

    def delete(self, identifier, namespace=None):
        self._ensure_not_empty(identifier=identifier)
        url = '/code_sources/%s' % identifier
        if namespace:
            url = url + '?namespace=%s' % namespace
        self._delete(url)