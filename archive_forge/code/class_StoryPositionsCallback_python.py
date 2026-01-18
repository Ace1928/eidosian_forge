from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class StoryPositionsCallback_python(StoryPositionsCallback):

    def __init__(self, python_callback):
        super().__init__()
        self.python_callback = python_callback

    def call(self, position):
        self.python_callback(position)