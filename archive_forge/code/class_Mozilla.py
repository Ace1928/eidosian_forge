import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
class Mozilla(UnixBrowser):
    """Launcher class for Mozilla browsers."""
    remote_args = ['%action', '%s']
    remote_action = ''
    remote_action_newwin = '-new-window'
    remote_action_newtab = '-new-tab'
    background = True