import suds
import suds.sax.element
import suds.sax.parser
import datetime
import os
import shutil
import tempfile
from logging import getLogger
@staticmethod
def __remove_default_location():
    """
        Removes the default cache location folder.

        This removal may be disabled by setting the
        remove_default_location_on_exit FileCache class attribute to False.

        """
    if FileCache.remove_default_location_on_exit:
        shutil.rmtree(FileCache.__default_location, ignore_errors=True)