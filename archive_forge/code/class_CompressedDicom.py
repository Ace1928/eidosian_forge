import sys
import os
import struct
import logging
import numpy as np
class CompressedDicom(RuntimeError):
    pass