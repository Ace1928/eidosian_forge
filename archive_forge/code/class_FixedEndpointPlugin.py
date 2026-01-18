from keystoneauth1 import discover
class FixedEndpointPlugin(BaseAuthPlugin):
    """A base class for plugins that have one fixed endpoint."""

    def __init__(self, endpoint=None):
        super(FixedEndpointPlugin, self).__init__()
        self.endpoint = endpoint

    def get_endpoint(self, session, **kwargs):
        """Return the supplied endpoint.

        Using this plugin the same endpoint is returned regardless of the
        parameters passed to the plugin. endpoint_override overrides the
        endpoint specified when constructing the plugin.
        """
        return kwargs.get('endpoint_override') or self.endpoint

    def get_endpoint_data(self, session, endpoint_override=None, discover_versions=True, **kwargs):
        """Return a valid endpoint data for a the service.

        :param session: A session object that can be used for communication.
        :type session: keystoneauth1.session.Session
        :param str endpoint_override: URL to use for version discovery.
        :param bool discover_versions: Whether to get version metadata from
                                       the version discovery document even
                                       if it major api version info can be
                                       inferred from the url.
                                       (optional, defaults to True)
        :param kwargs: Ignored.

        :raises keystoneauth1.exceptions.http.HttpError: An error from an
                                                         invalid HTTP response.

        :return: Valid EndpointData or None if not available.
        :rtype: `keystoneauth1.discover.EndpointData` or None
        """
        return super(FixedEndpointPlugin, self).get_endpoint_data(session, endpoint_override=endpoint_override or self.endpoint, discover_versions=discover_versions, **kwargs)