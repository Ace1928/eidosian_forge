import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def _create_args_to_params(self, node, name, size, image, location=None, networks=None, project=None, diskoffering=None, ex_keyname=None, ex_userdata=None, ex_security_groups=None, ex_displayname=None, ex_ip_address=None, ex_start_vm=False, ex_rootdisksize=None, ex_affinity_groups=None):
    server_params = {}
    if name:
        server_params['name'] = name
    if ex_displayname:
        server_params['displayname'] = ex_displayname
    if size:
        server_params['serviceofferingid'] = size.id
    if image:
        server_params['templateid'] = image.id
    if location:
        server_params['zoneid'] = location.id
    else:
        server_params['zoneid'] = self.list_locations()[0].id
    if networks:
        networks = ','.join([str(network.id) for network in networks])
        server_params['networkids'] = networks
    if project:
        server_params['projectid'] = project.id
    if diskoffering:
        server_params['diskofferingid'] = diskoffering.id
    if ex_keyname:
        server_params['keypair'] = ex_keyname
    if ex_userdata:
        ex_userdata = base64.b64encode(b(ex_userdata)).decode('ascii')
        server_params['userdata'] = ex_userdata
    if ex_security_groups:
        ex_security_groups = ','.join(ex_security_groups)
        server_params['securitygroupnames'] = ex_security_groups
    if ex_ip_address:
        server_params['ipaddress'] = ex_ip_address
    if ex_rootdisksize:
        server_params['rootdisksize'] = ex_rootdisksize
    if ex_start_vm is not None:
        server_params['startvm'] = ex_start_vm
    if ex_affinity_groups:
        affinity_group_ids = ','.join((ag.id for ag in ex_affinity_groups))
        server_params['affinitygroupids'] = affinity_group_ids
    return server_params