import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def _get_namespace_fixture(ns_name, rt_name=RESOURCE_TYPE1, **kwargs):
    ns = {'display_name': 'Flavor Quota', 'description': 'DESCRIPTION1', 'self': '/v2/metadefs/namespaces/%s' % ns_name, 'namespace': ns_name, 'visibility': 'public', 'protected': True, 'owner': 'admin', 'resource_types': [{'name': rt_name}], 'schema': '/v2/schemas/metadefs/namespace', 'created_at': '2014-08-14T09:07:06Z', 'updated_at': '2014-08-14T09:07:06Z'}
    ns.update(kwargs)
    return ns