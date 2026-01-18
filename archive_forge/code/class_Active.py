import os
import ovs.util
import ovs.vlog
class Active(object):
    name = 'ACTIVE'
    is_connected = True

    @staticmethod
    def deadline(fsm, now):
        if fsm.probe_interval:
            base = max(fsm.last_activity, fsm.state_entered)
            expiration = base + fsm.probe_interval
            if now < expiration or fsm.last_receive_attempt is None or fsm.last_receive_attempt >= expiration:
                return expiration
            else:
                return now + 1
        return None

    @staticmethod
    def run(fsm, now):
        vlog.dbg('%s: idle %d ms, sending inactivity probe' % (fsm.name, now - max(fsm.last_activity, fsm.state_entered)))
        fsm._transition(now, Reconnect.Idle)
        return PROBE