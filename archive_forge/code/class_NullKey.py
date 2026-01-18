from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
class NullKey:
    """A dummy key that ignores events passed to it by other keys

    A NullKey instance is the left key instance used by default
    for the left most keyboard key.

    """

    def _right_white_down(self):
        pass

    def _right_white_up(self):
        pass

    def _right_black_down(self):
        pass

    def _right_black_up(self):
        pass