import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
def _get_attrs_for_create(client_manager, attrs, parsed_args):
    client = client_manager.network
    if parsed_args.branching_points:
        attrs['port_chains'] = {}
        src_chain = None
        for c in parsed_args.branching_points:
            if ':' not in c:
                raise exceptions.CommandError('Error: You must specify at least one destination chain for each source chain.')
            colon_split = c.split(':')
            src_chain = colon_split.pop(0)
            sc_ = client.find_sfc_port_chain(src_chain, ignore_missing=False)['id']
            for i in colon_split:
                comma_split = i.split(',')
                unique = set(comma_split)
                if len(unique) != len(comma_split):
                    raise exceptions.CommandError('Error: Duplicate destination chains from source chain {}'.format(src_chain))
                if sc_ in attrs['port_chains']:
                    raise exceptions.CommandError('Error: Source chain {} is in use already '.format(src_chain))
                attrs['port_chains'][sc_] = []
                _validate_destination_chains(comma_split, attrs, client, sc_)