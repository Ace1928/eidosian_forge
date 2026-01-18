import os
import re
import socket  # for gethostname
import gyp.easy_xml as easy_xml
def _FindCommandInPath(command):
    """If there are no slashes in the command given, this function
     searches the PATH env to find the given command, and converts it
     to an absolute path.  We have to do this because MSVS is looking
     for an actual file to launch a debugger on, not just a command
     line.  Note that this happens at GYP time, so anything needing to
     be built needs to have a full path."""
    if '/' in command or '\\' in command:
        return command
    else:
        paths = os.environ.get('PATH', '').split(os.pathsep)
        for path in paths:
            item = os.path.join(path, command)
            if os.path.isfile(item) and os.access(item, os.X_OK):
                return item
    return command