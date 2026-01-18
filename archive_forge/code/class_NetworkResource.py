from openstack import resource
class NetworkResource(resource.Resource):
    revision_number = resource.Body('revision_number', type=int)
    _allow_unknown_attrs_in_body = True

    def _prepare_request(self, requires_id=None, prepend_key=False, patch=False, base_path=None, params=None, if_revision=None, **kwargs):
        req = super(NetworkResource, self)._prepare_request(requires_id=requires_id, prepend_key=prepend_key, patch=patch, base_path=base_path, params=params)
        if if_revision is not None:
            req.headers['If-Match'] = 'revision_number=%d' % if_revision
        return req