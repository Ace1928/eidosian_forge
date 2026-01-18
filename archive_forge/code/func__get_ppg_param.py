import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
def _get_ppg_param(attrs, ppg):
    attrs['port_pair_group_parameters'] = {}
    for key, value in ppg.items():
        if key == 'lb-fields':
            attrs['port_pair_group_parameters']['lb_fields'] = [field for field in value.split('&') if field]
        else:
            attrs['port_pair_group_parameters'][key] = value
    return attrs['port_pair_group_parameters']