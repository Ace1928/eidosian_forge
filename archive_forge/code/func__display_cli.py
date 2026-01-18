import datetime
import errno
import html
import json
import os
import random
import shlex
import textwrap
import time
from tensorboard import manager
def _display_cli(port, height, display_handle):
    del height
    del display_handle
    message = 'Please visit http://localhost:%d in a web browser.' % port
    print(message)