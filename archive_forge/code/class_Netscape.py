import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
class Netscape(UnixBrowser):
    """Launcher class for Netscape browser."""
    raise_opts = ['-noraise', '-raise']
    remote_args = ['-remote', 'openURL(%s%action)']
    remote_action = ''
    remote_action_newwin = ',new-window'
    remote_action_newtab = ',new-tab'
    background = True