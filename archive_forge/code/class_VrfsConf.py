import abc
import logging
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import ConfWithId
from os_ken.services.protocols.bgp.rtconf.base import ConfWithIdListener
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStatsListener
from os_ken.services.protocols.bgp.rtconf.base import MAX_NUM_EXPORT_RT
from os_ken.services.protocols.bgp.rtconf.base import MAX_NUM_IMPORT_RT
from os_ken.services.protocols.bgp.rtconf.base import MULTI_EXIT_DISC
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.base import SITE_OF_ORIGINS
from os_ken.services.protocols.bgp.rtconf.base import validate
from os_ken.services.protocols.bgp.rtconf.base import validate_med
from os_ken.services.protocols.bgp.rtconf.base import validate_soo_list
class VrfsConf(BaseConf):
    """Container for all VRF configurations."""
    ADD_VRF_CONF_EVT, REMOVE_VRF_CONF_EVT = range(2)
    VALID_EVT = frozenset([ADD_VRF_CONF_EVT, REMOVE_VRF_CONF_EVT])

    def __init__(self):
        super(VrfsConf, self).__init__()
        self._vrfs_by_rd_rf = {}
        self._vrfs_by_id = {}

    def _init_opt_settings(self, **kwargs):
        pass

    @property
    def vrf_confs(self):
        """Returns a list of configured `VrfConf`s
        """
        return list(self._vrfs_by_rd_rf.values())

    @property
    def vrf_interested_rts(self):
        interested_rts = set()
        for vrf_conf in self._vrfs_by_id.values():
            interested_rts.update(vrf_conf.import_rts)
        return interested_rts

    def update(self, **kwargs):
        raise NotImplementedError('Use either add/remove_vrf_conf methods instead.')

    def add_vrf_conf(self, vrf_conf):
        if vrf_conf.rd_rf_id in self._vrfs_by_rd_rf.keys():
            raise RuntimeConfigError(desc='VrfConf with rd_rf %s already exists' % str(vrf_conf.rd_rf_id))
        if vrf_conf.id in self._vrfs_by_id:
            raise RuntimeConfigError(desc='VrfConf with id %s already exists' % str(vrf_conf.id))
        self._vrfs_by_rd_rf[vrf_conf.rd_rf_id] = vrf_conf
        self._vrfs_by_id[vrf_conf.id] = vrf_conf
        self._notify_listeners(VrfsConf.ADD_VRF_CONF_EVT, vrf_conf)

    def remove_vrf_conf(self, route_dist=None, vrf_id=None, vrf_rf=None):
        """Removes any matching `VrfConf` for given `route_dist` or `vrf_id`

        Parameters:
            - `route_dist`: (str) route distinguisher of a configured VRF
            - `vrf_id`: (str) vrf ID
            - `vrf_rf`: (str) route family of the VRF configuration
        If only `route_dist` is given, removes `VrfConf`s for all supported
        address families for this `route_dist`. If `vrf_rf` is given, than only
        removes `VrfConf` for that specific route family. If only `vrf_id` is
        given, matching `VrfConf` will be removed.
        """
        if route_dist is None and vrf_id is None:
            raise RuntimeConfigError(desc='To delete supply route_dist or id.')
        vrf_rfs = SUPPORTED_VRF_RF
        if vrf_rf:
            vrf_rfs = vrf_rf
        removed_vrf_confs = []
        for route_family in vrf_rfs:
            if route_dist is not None:
                rd_rf_id = VrfConf.create_rd_rf_id(route_dist, route_family)
                vrf_conf = self._vrfs_by_rd_rf.pop(rd_rf_id, None)
                if vrf_conf:
                    self._vrfs_by_id.pop(vrf_conf.id, None)
                    removed_vrf_confs.append(vrf_conf)
            else:
                vrf_conf = self._vrfs_by_id.pop(vrf_id, None)
                if vrf_conf:
                    self._vrfs_by_rd_rf.pop(vrf_conf.rd_rd_id, None)
                    removed_vrf_confs.append(vrf_conf)
        for vrf_conf in removed_vrf_confs:
            self._notify_listeners(VrfsConf.REMOVE_VRF_CONF_EVT, vrf_conf)
        return removed_vrf_confs

    def get_vrf_conf(self, route_dist, vrf_rf, vrf_id=None):
        if route_dist is None and vrf_id is None:
            raise RuntimeConfigError(desc='To get VRF supply route_dist or vrf_id.')
        if route_dist is not None and vrf_id is not None:
            vrf1 = self._vrfs_by_id.get(vrf_id)
            rd_rf_id = VrfConf.create_rd_rf_id(route_dist, vrf_rf)
            vrf2 = self._vrfs_by_rd_rf.get(rd_rf_id)
            if vrf1 is not vrf2:
                raise RuntimeConfigError(desc='Given VRF ID (%s) and RD (%s) are not of same VRF.' % (vrf_id, route_dist))
            vrf = vrf1
        elif route_dist is not None:
            rd_rf_id = VrfConf.create_rd_rf_id(route_dist, vrf_rf)
            vrf = self._vrfs_by_rd_rf.get(rd_rf_id)
        else:
            vrf = self._vrfs_by_id.get(vrf_id)
        return vrf

    @property
    def vrfs_by_rd_rf_id(self):
        return dict(self._vrfs_by_rd_rf)

    @classmethod
    def get_valid_evts(cls):
        self_valid_evts = super(VrfsConf, cls).get_valid_evts()
        self_valid_evts.update(VrfsConf.VALID_EVT)
        return self_valid_evts

    def __repr__(self):
        return '<%s(%r)>' % (self.__class__.__name__, self._vrfs_by_id)

    @property
    def settings(self):
        return [vrf.settings for vrf in self._vrfs_by_id.values()]