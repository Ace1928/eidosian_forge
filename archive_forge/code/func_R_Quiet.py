import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
@R_Quiet.setter
def R_Quiet(self, value) -> None:
    ...