import time
import threading
class TimeUtils(object):

    def time(self):
        """Get the current time back

        :rtype: float
        :returns: The current time in seconds
        """
        return time.time()

    def sleep(self, value):
        """Sleep for a designated time

        :type value: float
        :param value: The time to sleep for in seconds
        """
        return time.sleep(value)