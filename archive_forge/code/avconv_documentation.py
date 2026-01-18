import subprocess as sp
import numpy as np
from .abstract import VideoReaderAbstract, VideoWriterAbstract
from .avprobe import avprobe
from .. import _AVCONV_APPLICATION
from .. import _AVCONV_PATH
from .. import _HAS_AVCONV
from ..utils import *
Writes frames using libav

    Using libav as a backend, this class
    provides sane initializations for the default case.
    