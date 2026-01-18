import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
class EventData(object):
    """ Collection of relevant data related to the ongoing transition attempt.

    Attributes:
        state (State): The State from which the Event was triggered.
        event (Event): The triggering Event.
        machine (Machine): The current Machine instance.
        model (object): The model/object the machine is bound to.
        args (list): Optional positional arguments from trigger method
            to store internally for possible later use.
        kwargs (dict): Optional keyword arguments from trigger method
            to store internally for possible later use.
        transition (Transition): Currently active transition. Will be assigned during triggering.
        error (Exception): In case a triggered event causes an Error, it is assigned here and passed on.
        result (bool): True in case a transition has been successful, False otherwise.
    """

    def __init__(self, state, event, machine, model, args, kwargs):
        """
        Args:
            state (State): The State from which the Event was triggered.
            event (Event): The triggering Event.
            machine (Machine): The current Machine instance.
            model (object): The model/object the machine is bound to.
            args (tuple): Optional positional arguments from trigger method
                to store internally for possible later use.
            kwargs (dict): Optional keyword arguments from trigger method
                to store internally for possible later use.
        """
        self.state = state
        self.event = event
        self.machine = machine
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.transition = None
        self.error = None
        self.result = False

    def update(self, state):
        """ Updates the EventData object with the passed state.

        Attributes:
            state (State, str or Enum): The state object, enum member or string to assign to EventData.
        """
        if not isinstance(state, State):
            self.state = self.machine.get_state(state)

    def __repr__(self):
        return "<%s('%s', %s)@%s>" % (type(self).__name__, self.state, getattr(self, 'transition'), id(self))