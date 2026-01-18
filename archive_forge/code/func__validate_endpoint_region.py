import flask_restful
import http.client
from keystone.api._shared import json_home_relations
from keystone.catalog import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
from keystone import exception
from keystone import notifications
from keystone.server import flask as ks_flask
@staticmethod
def _validate_endpoint_region(endpoint):
    """Ensure the region for the endpoint exists.

        If 'region_id' is used to specify the region, then we will let the
        manager/driver take care of this.  If, however, 'region' is used,
        then for backward compatibility, we will auto-create the region.

        """
    if endpoint.get('region_id') is None and endpoint.get('region') is not None:
        endpoint['region_id'] = endpoint.pop('region')
        try:
            PROVIDERS.catalog_api.get_region(endpoint['region_id'])
        except exception.RegionNotFound:
            region = dict(id=endpoint['region_id'])
            PROVIDERS.catalog_api.create_region(region, initiator=notifications.build_audit_initiator())
    return endpoint