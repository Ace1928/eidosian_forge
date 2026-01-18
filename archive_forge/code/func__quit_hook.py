import pygame.event
import pygame.display
from pygame import error, register_quit
from pygame.event import Event
def _quit_hook():
    """
    Hook that gets run to quit module
    """
    global _ft_init
    _ft_init = False