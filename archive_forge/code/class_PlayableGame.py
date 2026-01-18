from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import gym.error
from gym import Env, logger
from gym.core import ActType, ObsType
from gym.error import DependencyNotInstalled
from gym.logger import deprecation
class PlayableGame:
    """Wraps an environment allowing keyboard inputs to interact with the environment."""

    def __init__(self, env: Env, keys_to_action: Optional[Dict[Tuple[int, ...], int]]=None, zoom: Optional[float]=None):
        """Wraps an environment with a dictionary of keyboard buttons to action and if to zoom in on the environment.

        Args:
            env: The environment to play
            keys_to_action: The dictionary of keyboard tuples and action value
            zoom: If to zoom in on the environment render
        """
        if env.render_mode not in {'rgb_array', 'rgb_array_list'}:
            logger.error(f'PlayableGame wrapper works only with rgb_array and rgb_array_list render modes, but your environment render_mode = {env.render_mode}.')
        self.env = env
        self.relevant_keys = self._get_relevant_keys(keys_to_action)
        self.video_size = self._get_video_size(zoom)
        self.screen = pygame.display.set_mode(self.video_size)
        self.pressed_keys = []
        self.running = True

    def _get_relevant_keys(self, keys_to_action: Optional[Dict[Tuple[int], int]]=None) -> set:
        if keys_to_action is None:
            if hasattr(self.env, 'get_keys_to_action'):
                keys_to_action = self.env.get_keys_to_action()
            elif hasattr(self.env.unwrapped, 'get_keys_to_action'):
                keys_to_action = self.env.unwrapped.get_keys_to_action()
            else:
                raise MissingKeysToAction(f'{self.env.spec.id} does not have explicit key to action mapping, please specify one manually')
        assert isinstance(keys_to_action, dict)
        relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
        return relevant_keys

    def _get_video_size(self, zoom: Optional[float]=None) -> Tuple[int, int]:
        rendered = self.env.render()
        if isinstance(rendered, List):
            rendered = rendered[-1]
        assert rendered is not None and isinstance(rendered, np.ndarray)
        video_size = (rendered.shape[1], rendered.shape[0])
        if zoom is not None:
            video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))
        return video_size

    def process_event(self, event: Event):
        """Processes a PyGame event.

        In particular, this function is used to keep track of which buttons are currently pressed
        and to exit the :func:`play` function when the PyGame window is closed.

        Args:
            event: The event to process
        """
        if event.type == pygame.KEYDOWN:
            if event.key in self.relevant_keys:
                self.pressed_keys.append(event.key)
            elif event.key == pygame.K_ESCAPE:
                self.running = False
        elif event.type == pygame.KEYUP:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
        elif event.type == pygame.QUIT:
            self.running = False
        elif event.type == VIDEORESIZE:
            self.video_size = event.size
            self.screen = pygame.display.set_mode(self.video_size)