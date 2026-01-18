import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
class HierarchicalEmitter(BaseEventHooks):

    def __init__(self):
        self._lookup_cache = {}
        self._handlers = _PrefixTrie()
        self._unique_id_handlers = {}

    def _emit(self, event_name, kwargs, stop_on_response=False):
        """
        Emit an event with optional keyword arguments.

        :type event_name: string
        :param event_name: Name of the event
        :type kwargs: dict
        :param kwargs: Arguments to be passed to the handler functions.
        :type stop_on_response: boolean
        :param stop_on_response: Whether to stop on the first non-None
                                response. If False, then all handlers
                                will be called. This is especially useful
                                to handlers which mutate data and then
                                want to stop propagation of the event.
        :rtype: list
        :return: List of (handler, response) tuples from all processed
                 handlers.
        """
        responses = []
        handlers_to_call = self._lookup_cache.get(event_name)
        if handlers_to_call is None:
            handlers_to_call = self._handlers.prefix_search(event_name)
            self._lookup_cache[event_name] = handlers_to_call
        elif not handlers_to_call:
            return []
        kwargs['event_name'] = event_name
        responses = []
        for handler in handlers_to_call:
            logger.debug('Event %s: calling handler %s', event_name, handler)
            response = handler(**kwargs)
            responses.append((handler, response))
            if stop_on_response and response is not None:
                return responses
        return responses

    def emit(self, event_name, **kwargs):
        """
        Emit an event by name with arguments passed as keyword args.

            >>> responses = emitter.emit(
            ...     'my-event.service.operation', arg1='one', arg2='two')

        :rtype: list
        :return: List of (handler, response) tuples from all processed
                 handlers.
        """
        return self._emit(event_name, kwargs)

    def emit_until_response(self, event_name, **kwargs):
        """
        Emit an event by name with arguments passed as keyword args,
        until the first non-``None`` response is received. This
        method prevents subsequent handlers from being invoked.

            >>> handler, response = emitter.emit_until_response(
                'my-event.service.operation', arg1='one', arg2='two')

        :rtype: tuple
        :return: The first (handler, response) tuple where the response
                 is not ``None``, otherwise (``None``, ``None``).
        """
        responses = self._emit(event_name, kwargs, stop_on_response=True)
        if responses:
            return responses[-1]
        else:
            return (None, None)

    def _register(self, event_name, handler, unique_id=None, unique_id_uses_count=False):
        self._register_section(event_name, handler, unique_id, unique_id_uses_count, section=_MIDDLE)

    def _register_first(self, event_name, handler, unique_id=None, unique_id_uses_count=False):
        self._register_section(event_name, handler, unique_id, unique_id_uses_count, section=_FIRST)

    def _register_last(self, event_name, handler, unique_id, unique_id_uses_count=False):
        self._register_section(event_name, handler, unique_id, unique_id_uses_count, section=_LAST)

    def _register_section(self, event_name, handler, unique_id, unique_id_uses_count, section):
        if unique_id is not None:
            if unique_id in self._unique_id_handlers:
                count = self._unique_id_handlers[unique_id].get('count', None)
                if unique_id_uses_count:
                    if not count:
                        raise ValueError('Initial registration of  unique id %s was specified to use a counter. Subsequent register calls to unique id must specify use of a counter as well.' % unique_id)
                    else:
                        self._unique_id_handlers[unique_id]['count'] += 1
                elif count:
                    raise ValueError('Initial registration of unique id %s was specified to not use a counter. Subsequent register calls to unique id must specify not to use a counter as well.' % unique_id)
                return
            else:
                self._handlers.append_item(event_name, handler, section=section)
                unique_id_handler_item = {'handler': handler}
                if unique_id_uses_count:
                    unique_id_handler_item['count'] = 1
                self._unique_id_handlers[unique_id] = unique_id_handler_item
        else:
            self._handlers.append_item(event_name, handler, section=section)
        self._lookup_cache = {}

    def unregister(self, event_name, handler=None, unique_id=None, unique_id_uses_count=False):
        if unique_id is not None:
            try:
                count = self._unique_id_handlers[unique_id].get('count', None)
            except KeyError:
                return
            if unique_id_uses_count:
                if count is None:
                    raise ValueError('Initial registration of unique id %s was specified to use a counter. Subsequent unregister calls to unique id must specify use of a counter as well.' % unique_id)
                elif count == 1:
                    handler = self._unique_id_handlers.pop(unique_id)['handler']
                else:
                    self._unique_id_handlers[unique_id]['count'] -= 1
                    return
            else:
                if count:
                    raise ValueError('Initial registration of unique id %s was specified to not use a counter. Subsequent unregister calls to unique id must specify not to use a counter as well.' % unique_id)
                handler = self._unique_id_handlers.pop(unique_id)['handler']
        try:
            self._handlers.remove_item(event_name, handler)
            self._lookup_cache = {}
        except ValueError:
            pass

    def __copy__(self):
        new_instance = self.__class__()
        new_state = self.__dict__.copy()
        new_state['_handlers'] = copy.copy(self._handlers)
        new_state['_unique_id_handlers'] = copy.copy(self._unique_id_handlers)
        new_instance.__dict__ = new_state
        return new_instance