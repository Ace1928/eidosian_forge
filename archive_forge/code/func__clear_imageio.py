import os
import sys
import pytest
def _clear_imageio():
    for key in list(sys.modules.keys()):
        if key.startswith('imageio'):
            del sys.modules[key]