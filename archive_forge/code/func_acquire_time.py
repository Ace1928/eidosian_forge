from pprint import pformat
from six import iteritems
import re
@acquire_time.setter
def acquire_time(self, acquire_time):
    """
        Sets the acquire_time of this V1LeaseSpec.
        acquireTime is a time when the current lease was acquired.

        :param acquire_time: The acquire_time of this V1LeaseSpec.
        :type: datetime
        """
    self._acquire_time = acquire_time