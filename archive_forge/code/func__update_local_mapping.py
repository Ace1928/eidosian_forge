import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _update_local_mapping(self, local, direct_maps):
    """Replace any {0}, {1} ... values with data from the assertion.

        :param local: local mapping reference that needs to be updated
        :type local: dict
        :param direct_maps: identity values used to update local
        :type direct_maps: keystone.federation.utils.DirectMaps

        Example local::

            {'user': {'name': '{0} {1}', 'email': '{2}'}}

        Example direct_maps::

            [['Bob'], ['Thompson'], ['bob@example.com']]

        :returns: new local mapping reference with replaced values.

        The expected return structure is::

            {'user': {'name': 'Bob Thompson', 'email': 'bob@example.org'}}

        :raises keystone.exception.DirectMappingError: when referring to a
            remote match from a local section of a rule

        """
    LOG.debug('direct_maps: %s', direct_maps)
    LOG.debug('local: %s', local)
    new = {}
    for k, v in local.items():
        if isinstance(v, dict):
            new_value = self._update_local_mapping(v, direct_maps)
        elif isinstance(v, list):
            new_value = [self._update_local_mapping(item, direct_maps) for item in v]
        else:
            try:
                new_value = v.format(*direct_maps)
            except IndexError:
                raise exception.DirectMappingError(mapping_id=self.mapping_id)
        new[k] = new_value
    return new