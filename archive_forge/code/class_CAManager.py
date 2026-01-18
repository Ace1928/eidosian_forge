import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class CAManager(base.BaseEntityManager):
    """Entity Manager for Secret entities"""

    def __init__(self, api):
        super(CAManager, self).__init__(api, 'cas')

    def get(self, ca_ref):
        """Retrieve an existing CA from Barbican

        :param str ca_ref: Full HATEOAS reference to a CA
        :returns: CA object retrieved from Barbican
        :rtype: :class:`barbicanclient.v1.cas.CA`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        LOG.debug('Getting ca - CA href: {0}'.format(ca_ref))
        base.validate_ref_and_return_uuid(ca_ref, 'CA')
        return CA(api=self._api, ca_ref=ca_ref)

    def list(self, limit=10, offset=0, name=None):
        """List CAs for the project

        This method uses the limit and offset parameters for paging,
        and also supports filtering.

        :param limit: Max number of CAs returned
        :param offset: Offset secrets to begin list
        :param name: Name filter for the list
        :returns: list of CA objects that satisfy the provided filter
            criteria.
        :rtype: list
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        LOG.debug('Listing CAs - offset {0} limit {1}'.format(offset, limit))
        params = {'limit': limit, 'offset': offset}
        if name:
            params['name'] = name
        response = self._api.get(self._entity, params=params)
        return [CA(api=self._api, ca_ref=s) for s in response.get('cas', [])]