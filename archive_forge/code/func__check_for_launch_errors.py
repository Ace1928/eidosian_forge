import atexit
import functools
import locale
import logging
import multiprocessing
import os
import traceback
import pathlib
import Pyro4.core
import argparse
from enum import IntEnum
import shutil
import socket
import struct
import collections
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
import uuid
import psutil
import Pyro4
from random import Random
from minerl.env import comms
import minerl.utils.process_watcher
def _check_for_launch_errors(line):
    if 'at org.lwjgl.opengl.Display.<clinit>' in line:
        raise RuntimeError("ERROR! MineRL could not detect an X Server, Monitor, or Virtual Monitor! \n\nIn order to run minerl environments WITHOUT A HEAD use a software renderer such as 'xvfb':\n\t\txvfb-run python3 <your_script.py>\n\t! NOTE: xvfb conflicts with NVIDIA-drivers! \n\t! To run headless MineRL on a system with NVIDIA-drivers, please start a \n\t! vnc server of your choosing and then `export DISPLAY=:<insert ur vnc server #>\n\nIf you're receiving this error and there is a monitor attached, make sure your current displayvariable is set correctly: \n\t\t DISPLAY=:0 python3 <your_script.py>\n\t! NOTE: For this to work your account must be logged on the physical monitor.\n\nIf none of these steps work, please complain in the discord!\nIf all else fails, JUST PUT THIS IN A DOCKER CONTAINER! :)")
    if 'Could not choose GLX13 config' in line:
        raise RuntimeError("ERROR! MineRL could not detect any OpenGL libraries on your machine! \nTo fix this please install an OpenGL supporting graphics driver.\n\nIF THIS IS A HEADLESS LINUX MACHINE we reccomend Mesa:\n\n\tOn Ubuntu: \n\t\tsudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev\n\n\tOn other distributions:\n\t\thttps://www.mesa3d.org/install.html\n\n\t (If this still isn't working you may have a conflicting NVIDIA driver.)\n\t (In which case you'll need to run minerl in a docker container)\n\n\nIF THIS IS NOT A HEADLESS MACHINE please install graphics drivers on your system!\n\nIf none of these steps work, please complain in discord!\nIf all else fails, JUST PUT THIS IN A DOCKER CONTAINER! :)")