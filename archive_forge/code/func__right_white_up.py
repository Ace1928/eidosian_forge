from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
def _right_white_up(self):
    """Signal that the adjacent white key has been released

        This method is for internal propagation of events between
        key instances.

        """
    self._state, source_rect = self.key_data.c_event_right_white_up[self._state]
    if source_rect is not None:
        self._source_rect = source_rect
        self.key_data.c_updates.add(self)