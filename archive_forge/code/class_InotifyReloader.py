import os
import os.path
import re
import sys
import time
import threading
class InotifyReloader(object):

    def __init__(self, extra_files=None, callback=None):
        raise ImportError('You must have the inotify module installed to use the inotify reloader')