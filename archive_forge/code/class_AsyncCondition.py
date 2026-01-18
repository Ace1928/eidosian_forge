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
class AsyncCondition(Condition):
    """ A helper class to await condition checks in the intended way. """

    async def check(self, event_data):
        """ Check whether the condition passes.
        Args:
            event_data (EventData): An EventData instance to pass to the
                condition (if event sending is enabled) or to extract arguments
                from (if event sending is disabled). Also contains the data
                model attached to the current machine which is used to invoke
                the condition.
        """
        func = event_data.machine.resolve_callable(self.func, event_data)
        res = func(event_data) if event_data.machine.send_event else func(*event_data.args, **event_data.kwargs)
        if inspect.isawaitable(res):
            return await res == self.target
        return res == self.target