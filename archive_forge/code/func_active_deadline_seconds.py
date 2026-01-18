from pprint import pformat
from six import iteritems
import re
@active_deadline_seconds.setter
def active_deadline_seconds(self, active_deadline_seconds):
    """
        Sets the active_deadline_seconds of this V1JobSpec.
        Specifies the duration in seconds relative to the startTime that the job
        may be active before the system tries to terminate it; value must be
        positive integer

        :param active_deadline_seconds: The active_deadline_seconds of this
        V1JobSpec.
        :type: int
        """
    self._active_deadline_seconds = active_deadline_seconds