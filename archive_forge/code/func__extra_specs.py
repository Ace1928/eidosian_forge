from openstack import exceptions
from openstack import resource
from openstack import utils
def _extra_specs(self, method, key=None, delete=False, extra_specs=None):
    extra_specs = extra_specs or {}
    for k, v in extra_specs.items():
        if not isinstance(v, str):
            raise ValueError('The value for %s (%s) must be a text string' % (k, v))
    if key is not None:
        url = utils.urljoin(self.base_path, self.id, 'extra_specs', key)
    else:
        url = utils.urljoin(self.base_path, self.id, 'extra_specs')
    kwargs = {}
    if extra_specs:
        kwargs['json'] = {'extra_specs': extra_specs}
    response = method(url, headers={}, **kwargs)
    exceptions.raise_from_response(response)
    return response.json() if not delete else None