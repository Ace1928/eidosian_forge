import errno
import os
import pwd
import shutil
import stat
import tempfile
class PasswdError(Exception):
    """Exception class for errors loading a password from a file."""