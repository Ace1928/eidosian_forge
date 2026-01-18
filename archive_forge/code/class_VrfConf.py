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
class VrfConf(ConfWithId, ConfWithStats):
    """Class that encapsulates configurations for one VRF."""
    VRF_CHG_EVT = 'vrf_chg_evt'
    VALID_EVT = frozenset([VRF_CHG_EVT])
    REQUIRED_SETTINGS = frozenset([ROUTE_DISTINGUISHER, IMPORT_RTS, EXPORT_RTS])
    OPTIONAL_SETTINGS = frozenset([VRF_NAME, MULTI_EXIT_DISC, SITE_OF_ORIGINS, VRF_RF, IMPORT_MAPS])

    def __init__(self, **kwargs):
        """Create an instance of VRF runtime configuration."""
        super(VrfConf, self).__init__(**kwargs)

    def _init_opt_settings(self, **kwargs):
        super(VrfConf, self)._init_opt_settings(**kwargs)
        med = kwargs.pop(MULTI_EXIT_DISC, None)
        if med and validate_med(med):
            self._settings[MULTI_EXIT_DISC] = med
        soos = kwargs.pop(SITE_OF_ORIGINS, None)
        if soos and validate_soo_list(soos):
            self._settings[SITE_OF_ORIGINS] = soos
        vrf_rf = kwargs.pop(VRF_RF, VRF_RF_IPV4)
        if vrf_rf and validate_vrf_rf(vrf_rf):
            self._settings[VRF_RF] = vrf_rf
        import_maps = kwargs.pop(IMPORT_MAPS, [])
        self._settings[IMPORT_MAPS] = import_maps

    @property
    def route_dist(self):
        return self._settings[ROUTE_DISTINGUISHER]

    @property
    def import_rts(self):
        return list(self._settings[IMPORT_RTS])

    @property
    def export_rts(self):
        return list(self._settings[EXPORT_RTS])

    @property
    def soo_list(self):
        soos = self._settings.get(SITE_OF_ORIGINS)
        if soos:
            soos = list(soos)
        else:
            soos = []
        return soos

    @property
    def multi_exit_disc(self):
        """Returns configured value of MED, else None.

        This configuration does not have default value.
        """
        return self._settings.get(MULTI_EXIT_DISC)

    @property
    def route_family(self):
        """Returns configured route family for this VRF

        This configuration does not change.
        """
        return self._settings.get(VRF_RF)

    @property
    def rd_rf_id(self):
        return VrfConf.create_rd_rf_id(self.route_dist, self.route_family)

    @property
    def import_maps(self):
        return self._settings.get(IMPORT_MAPS)

    @staticmethod
    def create_rd_rf_id(route_dist, route_family):
        return (route_dist, route_family)

    @staticmethod
    def vrf_rf_2_rf(vrf_rf):
        if vrf_rf == VRF_RF_IPV4:
            return RF_IPv4_UC
        elif vrf_rf == VRF_RF_IPV6:
            return RF_IPv6_UC
        elif vrf_rf == VRF_RF_L2_EVPN:
            return RF_L2_EVPN
        elif vrf_rf == VRF_RF_IPV4_FLOWSPEC:
            return RF_IPv4_FLOWSPEC
        elif vrf_rf == VRF_RF_IPV6_FLOWSPEC:
            return RF_IPv6_FLOWSPEC
        elif vrf_rf == VRF_RF_L2VPN_FLOWSPEC:
            return RF_L2VPN_FLOWSPEC
        else:
            raise ValueError('Unsupported VRF route family given %s' % vrf_rf)

    @staticmethod
    def rf_2_vrf_rf(route_family):
        if route_family == RF_IPv4_UC:
            return VRF_RF_IPV4
        elif route_family == RF_IPv6_UC:
            return VRF_RF_IPV6
        elif route_family == RF_L2_EVPN:
            return VRF_RF_L2_EVPN
        elif route_family == RF_IPv4_FLOWSPEC:
            return VRF_RF_IPV4_FLOWSPEC
        elif route_family == RF_IPv6_FLOWSPEC:
            return VRF_RF_IPV6_FLOWSPEC
        elif route_family == RF_L2VPN_FLOWSPEC:
            return VRF_RF_L2VPN_FLOWSPEC
        else:
            raise ValueError('No supported mapping for route family to vrf_route_family exists for %s' % route_family)

    @property
    def settings(self):
        """Returns a copy of current settings.

        As some of the attributes are themselves containers, we clone the
        settings to provide clones for those containers as well.
        """
        cloned_setting = self._settings.copy()
        cloned_setting[IMPORT_RTS] = self.import_rts
        cloned_setting[EXPORT_RTS] = self.export_rts
        cloned_setting[SITE_OF_ORIGINS] = self.soo_list
        return cloned_setting

    @classmethod
    def get_opt_settings(cls):
        self_confs = super(VrfConf, cls).get_opt_settings()
        self_confs.update(VrfConf.OPTIONAL_SETTINGS)
        return self_confs

    @classmethod
    def get_req_settings(cls):
        self_confs = super(VrfConf, cls).get_req_settings()
        self_confs.update(VrfConf.REQUIRED_SETTINGS)
        return self_confs

    @classmethod
    def get_valid_evts(cls):
        self_valid_evts = super(VrfConf, cls).get_valid_evts()
        self_valid_evts.update(VrfConf.VALID_EVT)
        return self_valid_evts

    def update(self, **kwargs):
        """Updates this `VrfConf` settings.

        Notifies listeners if any settings changed. Returns `True` if update
        was successful. This vrfs' route family, id and route dist settings
        cannot be updated/changed.
        """
        super(VrfConf, self).update(**kwargs)
        vrf_id = kwargs.get(ConfWithId.ID)
        vrf_rd = kwargs.get(ROUTE_DISTINGUISHER)
        vrf_rf = kwargs.get(VRF_RF)
        if vrf_id != self.id or vrf_rd != self.route_dist or vrf_rf != self.route_family:
            raise ConfigValueError(desc='id/route-distinguisher/route-family do not match configured value.')
        new_imp_rts, old_imp_rts = self._update_import_rts(**kwargs)
        export_rts_changed = self._update_export_rts(**kwargs)
        soos_list_changed = self._update_soo_list(**kwargs)
        med_changed = self._update_med(**kwargs)
        re_export_needed = export_rts_changed or soos_list_changed or med_changed
        import_maps = kwargs.get(IMPORT_MAPS, [])
        re_import_needed = self._update_importmaps(import_maps)
        if new_imp_rts is not None or old_imp_rts is not None or re_export_needed or re_import_needed:
            evt_value = (new_imp_rts, old_imp_rts, import_maps, re_export_needed, re_import_needed)
            self._notify_listeners(VrfConf.VRF_CHG_EVT, evt_value)
        return True

    def _update_import_rts(self, **kwargs):
        import_rts = kwargs.get(IMPORT_RTS)
        get_validator(IMPORT_RTS)(import_rts)
        curr_import_rts = set(self._settings[IMPORT_RTS])
        import_rts = set(import_rts)
        if not import_rts.symmetric_difference(curr_import_rts):
            return (None, None)
        new_import_rts = import_rts - curr_import_rts
        old_import_rts = curr_import_rts - import_rts
        self._settings[IMPORT_RTS] = import_rts
        return (new_import_rts, old_import_rts)

    def _update_export_rts(self, **kwargs):
        export_rts = kwargs.get(EXPORT_RTS)
        get_validator(EXPORT_RTS)(export_rts)
        curr_export_rts = set(self._settings[EXPORT_RTS])
        if curr_export_rts.symmetric_difference(export_rts):
            self._settings[EXPORT_RTS] = list(export_rts)
            return True
        return False

    def _update_soo_list(self, **kwargs):
        soo_list = kwargs.get(SITE_OF_ORIGINS, [])
        get_validator(SITE_OF_ORIGINS)(soo_list)
        curr_soos = set(self.soo_list)
        if curr_soos.symmetric_difference(soo_list):
            self._settings[SITE_OF_ORIGINS] = soo_list[:]
            return True
        return False

    def _update_med(self, **kwargs):
        multi_exit_disc = kwargs.get(MULTI_EXIT_DISC, None)
        if multi_exit_disc:
            get_validator(MULTI_EXIT_DISC)(multi_exit_disc)
        if multi_exit_disc != self.multi_exit_disc:
            self._settings[MULTI_EXIT_DISC] = multi_exit_disc
            return True
        return False

    def _update_importmaps(self, import_maps):
        if set(self._settings[IMPORT_MAPS]).symmetric_difference(import_maps):
            self._settings[IMPORT_MAPS] = import_maps
            return True
        return False

    def __repr__(self):
        return '<%s(route_dist: %r, import_rts: %r, export_rts: %r, soo_list: %r)>' % (self.__class__.__name__, self.route_dist, self.import_rts, self.export_rts, self.soo_list)

    def __str__(self):
        return 'VrfConf-%s' % self.route_dist