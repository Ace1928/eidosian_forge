from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def _load_unicode(self, path):
    import shutil
    fdir = str(FONTDIR)
    temp = os.path.join(fdir, path)
    pgfont = os.path.join(fdir, 'test_sans.ttf')
    shutil.copy(pgfont, temp)
    try:
        with open(temp, 'rb') as f:
            pass
    except FileNotFoundError:
        raise unittest.SkipTest('the path cannot be opened')
    try:
        pygame_font.Font(temp, 20)
    finally:
        os.remove(temp)