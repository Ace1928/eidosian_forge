import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
class InvalidBool:
    """To help test invalid bool values."""
    __bool__ = None