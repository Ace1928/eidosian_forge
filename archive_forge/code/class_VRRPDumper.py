from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.services.protocols.vrrp import event as vrrp_event
class VRRPDumper(app_manager.OSKenApp):

    def __init__(self, *args, **kwargs):
        super(VRRPDumper, self).__init__(*args, **kwargs)

    @handler.set_ev_cls(vrrp_event.EventVRRPStateChanged)
    def vrrp_state_changed_handler(self, ev):
        old_state = ev.old_state
        new_state = ev.new_state
        self.logger.info('state change %s: %s -> %s', ev.instance_name, old_state, new_state)
        if new_state == vrrp_event.VRRP_STATE_MASTER:
            self.logger.info('becomes master')
            if old_state is None:
                pass
            elif old_state == vrrp_event.VRRP_STATE_BACKUP:
                pass
        elif new_state == vrrp_event.VRRP_STATE_BACKUP:
            self.logger.info('becomes backup')
        elif new_state == vrrp_event.VRRP_STATE_INITIALIZE:
            if old_state is None:
                self.logger.info('initialized')
            else:
                self.logger.info('shutdowned')
        else:
            raise ValueError('invalid vrrp state %s' % new_state)