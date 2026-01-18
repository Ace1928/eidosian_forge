import atexit
import logging
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Optional, TextIO, Union
Stop the server and cleans up auto generated resources (e.g. data dir).