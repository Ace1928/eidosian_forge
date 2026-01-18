import suds
import suds.sax.element
import suds.sax.parser
import datetime
import os
import shutil
import tempfile
from logging import getLogger
def _getf(self, id):
    """Open a cached file with the given id for reading."""
    try:
        filename = self.__filename(id)
        self.__remove_if_expired(filename)
        return self.__open(filename, 'rb')
    except Exception:
        pass