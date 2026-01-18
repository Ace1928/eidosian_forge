import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _get_sec_group_list(sec_group_ids):
    search_opts['id'] = sec_group_ids
    return neutron_client.list_security_groups(**search_opts).get('security_groups', [])