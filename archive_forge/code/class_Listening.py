import os
import ovs.util
import ovs.vlog
class Listening(object):
    name = 'LISTENING'
    is_connected = False

    @staticmethod
    def deadline(fsm, now):
        return None

    @staticmethod
    def run(fsm, now):
        return None