import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc.v2.networking_bgpvpn import constants
from neutronclient.osc.v2.networking_bgpvpn import resource_association
def _transform_resource(self, data):
    """Transforms BGP VPN port association routes property

        That permits to easily format the command output with ListColumn
        formater and separate the two route types.

        {'routes':
            [
                {
                    'type': 'prefix',
                    'local_pref': 100,
                    'prefix': '8.8.8.0/27',
                },
                {
                    'type': 'prefix',
                    'local_pref': 42,
                    'prefix': '80.50.30.0/28',
                },
                {
                    'type': 'bgpvpn',
                    'local_pref': 50,
                    'bgpvpn': '157d72a9-9968-48e7-8087-6c9a9bc7a181',
                },
                {
                    'type': 'bgpvpn',
                    'bgpvpn': 'd5c7aaab-c7e8-48b3-85ca-a115c00d3603',
                },
            ],
        }

        to

        {
            'prefix_routes': [
                '8.8.8.0/27 (100)',
                '80.50.30.0/28 (42)',
            ],
            'bgpvpn_routes': [
                '157d72a9-9968-48e7-8087-6c9a9bc7a181 (50)',
                'd5c7aaab-c7e8-48b3-85ca-a115c00d3603',
            ],
        }
        """
    for route in data.get('routes', []):
        local_pref = ''
        if route.get('local_pref'):
            local_pref = ' (%d)' % route.get('local_pref')
        if route['type'] == 'prefix':
            data.setdefault('prefix_routes', []).append('%s%s' % (route['prefix'], local_pref))
        elif route['type'] == 'bgpvpn':
            data.setdefault('bgpvpn_routes', []).append('%s%s' % (route['bgpvpn_id'], local_pref))
        else:
            LOG.warning('Unknown route type %s (%s).', route['type'], route)
    data.pop('routes', None)
    return data