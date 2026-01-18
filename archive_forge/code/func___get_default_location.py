import suds
import suds.sax.element
import suds.sax.parser
import datetime
import os
import shutil
import tempfile
from logging import getLogger
@staticmethod
def __get_default_location():
    """
        Returns the current process's default cache location folder.

        The folder is determined lazily on first call.

        """
    if not FileCache.__default_location:
        tmp = tempfile.mkdtemp('suds-default-cache')
        FileCache.__default_location = tmp
        import atexit
        atexit.register(FileCache.__remove_default_location)
    return FileCache.__default_location