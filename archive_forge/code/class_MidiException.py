from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
class MidiException(Exception):
    """exception that pygame.midi functions and classes can raise
    MidiException(errno)
    """

    def __init__(self, value):
        super().__init__(value)
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)