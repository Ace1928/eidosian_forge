from openstack import resource
class Limits(resource.Resource):
    base_path = '/limits'
    resource_key = 'limits'
    allow_fetch = True
    _query_mapping = resource.QueryParameters('tenant_id')
    absolute = resource.Body('absolute', type=AbsoluteLimits)
    rate = resource.Body('rate', type=list, list_type=RateLimit)

    def fetch(self, session, requires_id=False, error_message=None, base_path=None, skip_cache=False, **params):
        """Get the Limits resource.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`

        :returns: A Limits instance
        :rtype: :class:`~openstack.compute.v2.limits.Limits`
        """
        return super(Limits, self).fetch(session=session, requires_id=requires_id, error_message=error_message, base_path=base_path, skip_cache=skip_cache, **params)