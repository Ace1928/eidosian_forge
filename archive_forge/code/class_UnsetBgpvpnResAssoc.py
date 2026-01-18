import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.networking_bgpvpn import constants
class UnsetBgpvpnResAssoc(SetBgpvpnResAssoc):
    """Unset BGP VPN resource association properties"""
    _action = 'unset'