import hashlib
import os
from jupyterlab_pygments import JupyterStyle  # type:ignore[import-untyped]
from pygments.style import Style
from traitlets import Type, Unicode, Union
from .base import Preprocessor
Compute the hash of a file.