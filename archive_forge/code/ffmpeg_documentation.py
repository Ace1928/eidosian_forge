import re
import sys
import time
import logging
import platform
import threading
import subprocess as sp
import imageio_ffmpeg
import numpy as np
from ..core import Format, image_as_uint
Reads and throws away n frames