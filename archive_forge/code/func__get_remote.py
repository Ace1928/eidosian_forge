import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _get_remote(rule):
    if rule['remote_ip_prefix']:
        remote = '%s (CIDR)' % rule['remote_ip_prefix']
    elif rule['remote_group_id']:
        remote = '%s (group)' % rule['remote_group_id']
    else:
        remote = None
    return remote