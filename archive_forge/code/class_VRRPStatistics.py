import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
class VRRPStatistics(object):

    def __init__(self, name, resource_id, statistics_interval):
        self.name = name
        self.resource_id = resource_id
        self.statistics_interval = statistics_interval
        self.tx_vrrp_packets = 0
        self.rx_vrrp_packets = 0
        self.rx_vrrp_zero_prio_packets = 0
        self.tx_vrrp_zero_prio_packets = 0
        self.rx_vrrp_invalid_packets = 0
        self.rx_vrrp_bad_auth = 0
        self.idle_to_master_transitions = 0
        self.idle_to_backup_transitions = 0
        self.backup_to_master_transitions = 0
        self.master_to_backup_transitions = 0

    def get_stats(self):
        ts = time.strftime('%Y-%m-%dT%H:%M:%S')
        stats_dict = dict(timestamp=ts, resource_id=self.resource_id, tx_vrrp_packets=self.tx_vrrp_packets, rx_vrrp_packets=self.rx_vrrp_packets, rx_vrrp_zero_prio_packets=self.rx_vrrp_zero_prio_packets, tx_vrrp_zero_prio_packets=self.tx_vrrp_zero_prio_packets, rx_vrrp_invalid_packets=self.rx_vrrp_invalid_packets, rx_vrrp_bad_auth=self.rx_vrrp_bad_auth, idle_to_master_transitions=self.idle_to_master_transitions, idle_to_backup_transitions=self.idle_to_backup_transitions, backup_to_master_transitions=self.backup_to_master_transitions, master_to_backup_transitions=self.master_to_backup_transitions)
        return stats_dict