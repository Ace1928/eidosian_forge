from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
class Retry(State):
    """ The Retry mix-in sets a limit on the number of times a state may be
        re-entered from itself.

        The first time a state is entered it does not count as a retry. Thus with
        `retries=3` the state can be entered four times before it fails.

        When the retry limit is exceeded, the state is not entered and instead the
        `on_failure` callback is invoked on the model. For example,

            Retry(retries=3, on_failure='to_failed')

        transitions the model directly to the 'failed' state, if the machine has
        automatic transitions enabled (the default).

        Attributes:
            retries (int): Number of retries to allow before failing.
            on_failure (str): Function to invoke on the model when the retry limit
                is exceeded.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            **kwargs: If kwargs contains `retries`, then limit the number of times
                the state may be re-entered from itself. The argument `on_failure`,
                which is the function to invoke on the model when the retry limit
                is exceeded, must also be provided.
        """
        self.retries = kwargs.pop('retries', 0)
        self.on_failure = kwargs.pop('on_failure', None)
        self.retry_counts = Counter()
        if self.retries > 0 and self.on_failure is None:
            raise AttributeError("Retry state requires 'on_failure' when 'retries' is set.")
        super(Retry, self).__init__(*args, **kwargs)

    def enter(self, event_data):
        k = id(event_data.model)
        if event_data.transition.source != self.name:
            _LOGGER.debug('%sRetry limit for state %s reset (came from %s)', event_data.machine.name, self.name, event_data.transition.source)
            self.retry_counts[k] = 0
        if self.retry_counts[k] > self.retries > 0:
            _LOGGER.info('%sRetry count for state %s exceeded limit (%i)', event_data.machine.name, self.name, self.retries)
            event_data.machine.callback(self.on_failure, event_data)
            return
        _LOGGER.debug('%sRetry count for state %s is now %i', event_data.machine.name, self.name, self.retry_counts[k])
        self.retry_counts.update((k,))
        super(Retry, self).enter(event_data)