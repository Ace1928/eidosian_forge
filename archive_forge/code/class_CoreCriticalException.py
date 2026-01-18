import os
import sysconfig
import sys
import traceback
import tempfile
import subprocess
import importlib
import kivy
from kivy.logger import Logger
class CoreCriticalException(Exception):
    pass