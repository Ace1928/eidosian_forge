import logging
import asyncio
import contextvars
import inspect
from collections import deque
from functools import partial, reduce
import copy
from ..core import State, Condition, Transition, EventData, listify
from ..core import Event, MachineError, Machine
from .nesting import HierarchicalMachine, NestedState, NestedEvent, NestedTransition, resolve_order
class AsyncState(State):
    """ A persistent representation of a state managed by a ``Machine``. Callback execution is done asynchronously. """

    async def enter(self, event_data):
        """ Triggered when a state is entered.
        Args:
            event_data: (AsyncEventData): The currently processed event.
        """
        _LOGGER.debug('%sEntering state %s. Processing callbacks...', event_data.machine.name, self.name)
        await event_data.machine.callbacks(self.on_enter, event_data)
        _LOGGER.info('%sFinished processing state %s enter callbacks.', event_data.machine.name, self.name)

    async def exit(self, event_data):
        """ Triggered when a state is exited.
        Args:
            event_data: (AsyncEventData): The currently processed event.
        """
        _LOGGER.debug('%sExiting state %s. Processing callbacks...', event_data.machine.name, self.name)
        await event_data.machine.callbacks(self.on_exit, event_data)
        _LOGGER.info('%sFinished processing state %s exit callbacks.', event_data.machine.name, self.name)