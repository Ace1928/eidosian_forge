from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
from ray.autoscaler._private.cli_logger import cli_logger
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