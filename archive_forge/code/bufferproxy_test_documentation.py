import re
import weakref
import gc
import ctypes
import unittest
import pygame
from pygame.bufferproxy import BufferProxy
refcount agnostic check that contained objects are freed