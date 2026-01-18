import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
def _format_zone(zone):
    zone.pop('links', None)
    zone['masters'] = ', '.join(zone['masters'])
    attrib = ''
    for attr in zone['attributes']:
        attrib += '{}:{}\n'.format(attr, zone['attributes'][attr])
    zone['attributes'] = attrib