import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config

        From dill._dill.save_code
        This is a modified version that removes the origin (filename + line no.)
        of functions created in notebooks or shells for example.
        