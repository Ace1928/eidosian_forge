import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
@classmethod
def _format_levelname(cls, levelname):
    if levelname in cls.LEVEL_COLORS:
        return cls.COLOR_SEQ % (30 + cls.LEVEL_COLORS[levelname]) + levelname + cls.RESET_SEQ
    return levelname