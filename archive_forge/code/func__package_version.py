import platform
import sys
from importlib.metadata import version
def _package_version(p):
    try:
        print(f'{p:20}:  {version(p)}')
    except ImportError:
        print(f'{p:20}:  -')