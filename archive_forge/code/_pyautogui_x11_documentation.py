import pyautogui
import sys
import os
import subprocess
from pyautogui import LEFT, MIDDLE, RIGHT
from Xlib.display import Display
from Xlib import X
from Xlib.ext.xtest import fake_input
import Xlib.XK

    Release a given character key. Also works with character keycodes as
    integers, but not keysyms.
    