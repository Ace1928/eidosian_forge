from openstack import resource
class Keypair(resource.Resource):
    resource_key = 'keypair'
    resources_key = 'keypairs'
    base_path = '/os-keypairs'
    _query_mapping = resource.QueryParameters('user_id')
    allow_create = True
    allow_fetch = True
    allow_delete = True
    allow_list = True
    _max_microversion = '2.10'
    created_at = resource.Body('created_at')
    is_deleted = resource.Body('deleted', type=bool)
    fingerprint = resource.Body('fingerprint')
    id = resource.Body('name')
    name = resource.Body('name', alternate_id=True)
    private_key = resource.Body('private_key')
    public_key = resource.Body('public_key')
    type = resource.Body('type', default='ssh')
    user_id = resource.Body('user_id')

    def _consume_attrs(self, mapping, attrs):
        if 'id' in attrs:
            attrs.setdefault('name', attrs.pop('id'))
        return super(Keypair, self)._consume_attrs(mapping, attrs)

    @classmethod
    def existing(cls, connection=None, **kwargs):
        """Create an instance of an existing remote resource.

        When creating the instance set the ``_synchronized`` parameter
        of :class:`Resource` to ``True`` to indicate that it represents the
        state of an existing server-side resource. As such, all attributes
        passed in ``**kwargs`` are considered "clean", such that an immediate
        :meth:`update` call would not generate a body of attributes to be
        modified on the server.

        :param dict kwargs: Each of the named arguments will be set as
                            attributes on the resulting Resource object.
        """
        if cls.resource_key in kwargs:
            args = kwargs.pop(cls.resource_key)
            kwargs.update(**args)
        return cls(_synchronized=True, connection=connection, **kwargs)