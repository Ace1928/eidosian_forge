import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def NextGreaterPowerOf2(x):
    return 2 ** x.bit_length()