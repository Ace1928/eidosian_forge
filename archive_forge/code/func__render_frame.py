import numpy as np
import gym
from gym.error import DependencyNotInstalled
def _render_frame(self):
    """Fetch the last frame from the base environment and render it to the screen."""
    try:
        import pygame
    except ImportError:
        raise DependencyNotInstalled('pygame is not installed, run `pip install gym[box2d]`')
    if self.env.render_mode == 'rgb_array_list':
        last_rgb_array = self.env.render()
        assert isinstance(last_rgb_array, list)
        last_rgb_array = last_rgb_array[-1]
    elif self.env.render_mode == 'rgb_array':
        last_rgb_array = self.env.render()
    else:
        raise Exception(f"Wrapped environment must have mode 'rgb_array' or 'rgb_array_list', actual render mode: {self.env.render_mode}")
    assert isinstance(last_rgb_array, np.ndarray)
    rgb_array = np.transpose(last_rgb_array, axes=(1, 0, 2))
    if self.screen_size is None:
        self.screen_size = rgb_array.shape[:2]
    assert self.screen_size == rgb_array.shape[:2], f'The shape of the rgb array has changed from {self.screen_size} to {rgb_array.shape[:2]}'
    if self.window is None:
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(self.screen_size)
    if self.clock is None:
        self.clock = pygame.time.Clock()
    surf = pygame.surfarray.make_surface(rgb_array)
    self.window.blit(surf, (0, 0))
    pygame.event.pump()
    self.clock.tick(self.metadata['render_fps'])
    pygame.display.flip()