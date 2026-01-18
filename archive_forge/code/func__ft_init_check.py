import pygame.event
import pygame.display
from pygame import error, register_quit
from pygame.event import Event
def _ft_init_check():
    """
    Raises error if module is not init
    """
    if not _ft_init:
        raise error('fastevent system not initialized')