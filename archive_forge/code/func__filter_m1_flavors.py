from openstack import exceptions
from openstack.tests.functional import base
def _filter_m1_flavors(self, results):
    """The m1 flavors are the original devstack flavors"""
    new_results = []
    for flavor in results:
        if flavor['name'].startswith('m1.'):
            new_results.append(flavor)
    return new_results