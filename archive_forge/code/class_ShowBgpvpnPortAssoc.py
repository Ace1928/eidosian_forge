import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc.v2.networking_bgpvpn import constants
from neutronclient.osc.v2.networking_bgpvpn import resource_association
class ShowBgpvpnPortAssoc(BgpvpnPortAssoc, resource_association.ShowBgpvpnResAssoc):
    _description = _('Show information of a given BGP VPN port association')