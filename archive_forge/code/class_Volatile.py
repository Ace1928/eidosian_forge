from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
class Volatile(State):
    """ Adds scopes/temporal variables to the otherwise persistent state objects.
    Attributes:
        volatile_cls (cls): Class of the temporal object to be initiated.
        volatile_hook (str): Model attribute name which will contain the volatile instance.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            **kwargs: If kwargs contains `volatile`, always create an instance of the passed class
                whenever the state is entered. The instance is assigned to a model attribute which
                can be passed with the kwargs keyword `hook`. If hook is not passed, the instance will
                be assigned to the 'attribute' scope. If `volatile` is not passed, an empty object will
                be assigned to the model's hook.
        """
        self.volatile_cls = kwargs.pop('volatile', VolatileObject)
        self.volatile_hook = kwargs.pop('hook', 'scope')
        super(Volatile, self).__init__(*args, **kwargs)
        self.initialized = True

    def enter(self, event_data):
        """ Extends `transitions.core.State.enter` by creating a volatile object and assign it to
            the current model's hook. """
        setattr(event_data.model, self.volatile_hook, self.volatile_cls())
        super(Volatile, self).enter(event_data)

    def exit(self, event_data):
        """ Extends `transitions.core.State.exit` by deleting the temporal object from the model. """
        super(Volatile, self).exit(event_data)
        try:
            delattr(event_data.model, self.volatile_hook)
        except AttributeError:
            pass