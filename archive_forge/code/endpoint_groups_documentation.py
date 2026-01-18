from keystoneclient import base
Delete an endpoint group.

        :param endpoint_group: the endpoint group to be deleted on the server.
        :type endpoint_group:
            str or :class:`keystoneclient.v3.endpoint_groups.EndpointGroup`

        :returns: Response object with 204 status.
        :rtype: :class:`requests.models.Response`

        