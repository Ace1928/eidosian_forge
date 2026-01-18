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
class AsyncTransition(Transition):
    """ Representation of an asynchronous transition managed by a ``AsyncMachine`` instance. """
    condition_cls = AsyncCondition

    async def _eval_conditions(self, event_data):
        res = await event_data.machine.await_all([partial(cond.check, event_data) for cond in self.conditions])
        if not all(res):
            _LOGGER.debug('%sTransition condition failed: Transition halted.', event_data.machine.name)
            return False
        return True

    async def execute(self, event_data):
        """ Executes the transition.
        Args:
            event_data (EventData): An instance of class EventData.
        Returns: boolean indicating whether or not the transition was
            successfully executed (True if successful, False if not).
        """
        _LOGGER.debug('%sInitiating transition from state %s to state %s...', event_data.machine.name, self.source, self.dest)
        await event_data.machine.callbacks(self.prepare, event_data)
        _LOGGER.debug('%sExecuted callbacks before conditions.', event_data.machine.name)
        if not await self._eval_conditions(event_data):
            return False
        machine = event_data.machine
        await machine.switch_model_context(event_data.model)
        await event_data.machine.callbacks(event_data.machine.before_state_change, event_data)
        await event_data.machine.callbacks(self.before, event_data)
        _LOGGER.debug('%sExecuted callback before transition.', event_data.machine.name)
        if self.dest:
            await self._change_state(event_data)
        await event_data.machine.callbacks(self.after, event_data)
        await event_data.machine.callbacks(event_data.machine.after_state_change, event_data)
        _LOGGER.debug('%sExecuted callback after transition.', event_data.machine.name)
        return True

    async def _change_state(self, event_data):
        if hasattr(event_data.machine, 'model_graphs'):
            graph = event_data.machine.model_graphs[id(event_data.model)]
            graph.reset_styling()
            graph.set_previous_transition(self.source, self.dest)
        await event_data.machine.get_state(self.source).exit(event_data)
        event_data.machine.set_state(self.dest, event_data.model)
        event_data.update(getattr(event_data.model, event_data.machine.model_attribute))
        await event_data.machine.get_state(self.dest).enter(event_data)