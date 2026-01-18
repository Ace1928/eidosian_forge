import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod
from pygame import error
def _pre_init_placeholder():
    if not _is_init:
        raise error('pygame.camera is not initialized')
    raise NotImplementedError()