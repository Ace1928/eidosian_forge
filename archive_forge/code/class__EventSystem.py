from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
from ray.autoscaler._private.cli_logger import cli_logger
class _EventSystem:
    """Event system that handles storing and calling callbacks for events.

    Attributes:
        callback_map (Dict[str, List[Callable]]) : Stores list of callbacks
            for events when registered.
    """

    def __init__(self):
        self.callback_map = {}

    def add_callback_handler(self, event: str, callback: Union[Callable[[Dict], None], List[Callable[[Dict], None]]]):
        """Stores callback handler for event.

        Args:
            event: Event that callback should be called on. See
                CreateClusterEvent for details on the events available to be
                registered against.
            callback (Callable[[Dict], None]): Callable object that is invoked
                when specified event occurs.
        """
        if event not in CreateClusterEvent.__members__.values():
            cli_logger.warning(f'{event} is not currently tracked, and this callback will not be invoked.')
        self.callback_map.setdefault(event, []).extend([callback] if type(callback) is not list else callback)

    def execute_callback(self, event: CreateClusterEvent, event_data: Optional[Dict[str, Any]]=None):
        """Executes all callbacks for event.

        Args:
            event: Event that is invoked. See CreateClusterEvent
                for details on the available events.
            event_data (Dict[str, Any]): Argument that is passed to each
                callable object stored for this particular event.
        """
        if event_data is None:
            event_data = {}
        event_data['event_name'] = event
        if event in self.callback_map:
            for callback in self.callback_map[event]:
                callback(event_data)

    def clear_callbacks_for_event(self, event: str):
        """Clears stored callable objects for event.

        Args:
            event: Event that has callable objects stored in map.
                See CreateClusterEvent for details on the available events.
        """
        if event in self.callback_map:
            del self.callback_map[event]