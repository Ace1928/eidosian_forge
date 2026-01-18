from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
class Timeout(State):
    """ Adds timeout functionality to a state. Timeouts are handled model-specific.
    Attributes:
        timeout (float): Seconds after which a timeout function should be called.
        on_timeout (list): Functions to call when a timeout is triggered.
    """
    dynamic_methods = ['on_timeout']

    def __init__(self, *args, **kwargs):
        """
        Args:
            **kwargs: If kwargs contain 'timeout', assign the float value to self.timeout. If timeout
                is set, 'on_timeout' needs to be passed with kwargs as well or an AttributeError will
                be thrown. If timeout is not passed or equal 0.
        """
        self.timeout = kwargs.pop('timeout', 0)
        self._on_timeout = None
        if self.timeout > 0:
            try:
                self.on_timeout = kwargs.pop('on_timeout')
            except KeyError:
                raise AttributeError("Timeout state requires 'on_timeout' when timeout is set.")
        else:
            self._on_timeout = kwargs.pop('on_timeout', [])
        self.runner = {}
        super(Timeout, self).__init__(*args, **kwargs)

    def enter(self, event_data):
        """ Extends `transitions.core.State.enter` by starting a timeout timer for the current model
            when the state is entered and self.timeout is larger than 0.
        """
        if self.timeout > 0:
            timer = Timer(self.timeout, self._process_timeout, args=(event_data,))
            timer.daemon = True
            timer.start()
            self.runner[id(event_data.model)] = timer
        return super(Timeout, self).enter(event_data)

    def exit(self, event_data):
        """ Extends `transitions.core.State.exit` by canceling a timer for the current model. """
        timer = self.runner.get(id(event_data.model), None)
        if timer is not None and timer.is_alive():
            timer.cancel()
        return super(Timeout, self).exit(event_data)

    def _process_timeout(self, event_data):
        _LOGGER.debug('%sTimeout state %s. Processing callbacks...', event_data.machine.name, self.name)
        for callback in self.on_timeout:
            event_data.machine.callback(callback, event_data)
        _LOGGER.info('%sTimeout state %s processed.', event_data.machine.name, self.name)

    @property
    def on_timeout(self):
        """ List of strings and callables to be called when the state timeouts. """
        return self._on_timeout

    @on_timeout.setter
    def on_timeout(self, value):
        """ Listifies passed values and assigns them to on_timeout."""
        self._on_timeout = listify(value)