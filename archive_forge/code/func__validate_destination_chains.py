import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
def _validate_destination_chains(comma_split, attrs, client, sc_):
    for e in comma_split:
        if e != '':
            dc_ = client.find_sfc_port_chain(e, ignore_missing=False)['id']
            attrs['port_chains'][sc_].append(dc_)
            if _check_cycle(attrs['port_chains'], sc_, dc_):
                raise exceptions.CommandError('Error: Service graph contains a cycle')
        else:
            raise exceptions.CommandError('Error: you must specify at least one destination chain for each source chain')
    return attrs