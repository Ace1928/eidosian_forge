import errno
import shlex
import subprocess
class FFExecutableNotFoundError(Exception):
    """Raise when FFmpeg/FFprobe executable was not found."""