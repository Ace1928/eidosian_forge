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
class HierarchicalAsyncMachine(HierarchicalMachine, AsyncMachine):
    """ Asynchronous variant of transitions.extensions.nesting.HierarchicalMachine.
        An asynchronous hierarchical machine REQUIRES AsyncNestedStates, AsyncNestedEvent and AsyncNestedTransitions
        (or any subclass of it) to operate.
    """
    state_cls = NestedAsyncState
    transition_cls = NestedAsyncTransition
    event_cls = NestedAsyncEvent

    async def trigger_event(self, model, trigger, *args, **kwargs):
        """ Processes events recursively and forwards arguments if suitable events are found.
        This function is usually bound to models with model and trigger arguments already
        resolved as a partial. Execution will halt when a nested transition has been executed
        successfully.
        Args:
            model (object): targeted model
            trigger (str): event name
            *args: positional parameters passed to the event and its callbacks
            **kwargs: keyword arguments passed to the event and its callbacks
        Returns:
            bool: whether a transition has been executed successfully
        Raises:
            MachineError: When no suitable transition could be found and ignore_invalid_trigger
                          is not True. Note that a transition which is not executed due to conditions
                          is still considered valid.
        """
        event_data = AsyncEventData(state=None, event=None, machine=self, model=model, args=args, kwargs=kwargs)
        event_data.result = None
        return await self.process_context(partial(self._trigger_event, event_data, trigger), model)

    async def _trigger_event(self, event_data, trigger):
        try:
            with self():
                res = await self._trigger_event_nested(event_data, trigger, None)
            event_data.result = self._check_event_result(res, event_data.model, trigger)
        except Exception as err:
            event_data.error = err
            if self.on_exception:
                await self.callbacks(self.on_exception, event_data)
            else:
                raise
        finally:
            try:
                await self.callbacks(self.finalize_event, event_data)
                _LOGGER.debug('%sExecuted machine finalize callbacks', self.name)
            except Exception as err:
                _LOGGER.error('%sWhile executing finalize callbacks a %s occurred: %s.', self.name, type(err).__name__, str(err))
        return event_data.result

    async def _trigger_event_nested(self, event_data, _trigger, _state_tree):
        model = event_data.model
        if _state_tree is None:
            _state_tree = self.build_state_tree(listify(getattr(model, self.model_attribute)), self.state_cls.separator)
        res = {}
        for key, value in _state_tree.items():
            if value:
                with self(key):
                    tmp = await self._trigger_event_nested(event_data, _trigger, value)
                    if tmp is not None:
                        res[key] = tmp
            if not res.get(key, None) and _trigger in self.events:
                tmp = await self.events[_trigger].trigger_nested(event_data)
                if tmp is not None:
                    res[key] = tmp
        return None if not res or all((v is None for v in res.values())) else any(res.values())

    async def _can_trigger(self, model, trigger, *args, **kwargs):
        state_tree = self.build_state_tree(getattr(model, self.model_attribute), self.state_cls.separator)
        ordered_states = resolve_order(state_tree)
        for state_path in ordered_states:
            with self():
                return await self._can_trigger_nested(model, trigger, state_path, *args, **kwargs)

    async def _can_trigger_nested(self, model, trigger, path, *args, **kwargs):
        evt = AsyncEventData(None, None, self, model, args, kwargs)
        if trigger in self.events:
            source_path = copy.copy(path)
            while source_path:
                state_name = self.state_cls.separator.join(source_path)
                for transition in self.events[trigger].transitions.get(state_name, []):
                    try:
                        _ = self.get_state(transition.dest)
                    except ValueError:
                        continue
                    await self.callbacks(self.prepare_event, evt)
                    await self.callbacks(transition.prepare, evt)
                    if all(await self.await_all([partial(c.check, evt) for c in transition.conditions])):
                        return True
                source_path.pop(-1)
        if path:
            with self(path.pop(0)):
                return await self._can_trigger_nested(model, trigger, path, *args, **kwargs)
        return False