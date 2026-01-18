import logging
from os_ken.controller import event
from os_ken.lib.dpid import dpid_to_str
from os_ken.base import app_manager
class ConfSwitchSet(app_manager.OSKenApp):

    def __init__(self):
        super(ConfSwitchSet, self).__init__()
        self.name = 'conf_switch'
        self.confs = {}

    def dpids(self):
        return list(self.confs.keys())

    def del_dpid(self, dpid):
        del self.confs[dpid]
        self.send_event_to_observers(EventConfSwitchDelDPID(dpid))

    def keys(self, dpid):
        return list(self.confs[dpid].keys())

    def set_key(self, dpid, key, value):
        conf = self.confs.setdefault(dpid, {})
        conf[key] = value
        self.send_event_to_observers(EventConfSwitchSet(dpid, key, value))

    def get_key(self, dpid, key):
        return self.confs[dpid][key]

    def del_key(self, dpid, key):
        del self.confs[dpid][key]
        self.send_event_to_observers(EventConfSwitchDel(dpid, key))

    def __contains__(self, item):
        """(dpid, key) in <ConfSwitchSet instance>"""
        dpid, key = item
        return dpid in self.confs and key in self.confs[dpid]

    def find_dpid(self, key, value):
        for dpid, conf in self.confs.items():
            if key in conf:
                if conf[key] == value:
                    return dpid
        return None