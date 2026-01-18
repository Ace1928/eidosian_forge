import inspect
import logging
import sys
class _Caller(object):
    """Describe how to handle an event class.
    """

    def __init__(self, dispatchers, ev_source):
        """Initialize _Caller.

        :param dispatchers: A list of states or a state, in which this
                            is in effect.
                            None and [] mean all states.
        :param ev_source: The module which generates the event.
                          ev_cls.__module__ for set_ev_cls.
                          None for set_ev_handler.
        """
        self.dispatchers = dispatchers
        self.ev_source = ev_source