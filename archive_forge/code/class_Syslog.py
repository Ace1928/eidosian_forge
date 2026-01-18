import py
import sys
class Syslog:
    """ consumer that writes to the syslog daemon """

    def __init__(self, priority=None):
        if priority is None:
            priority = self.LOG_INFO
        self.priority = priority

    def __call__(self, msg):
        """ write a message to the log """
        import syslog
        syslog.syslog(self.priority, str(msg))