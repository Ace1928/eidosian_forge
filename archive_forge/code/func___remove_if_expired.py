import suds
import suds.sax.element
import suds.sax.parser
import datetime
import os
import shutil
import tempfile
from logging import getLogger
def __remove_if_expired(self, filename):
    """
        Remove a cached file entry if it expired.

        @param filename: The file name.
        @type filename: str

        """
    if not self.duration:
        return
    created = datetime.datetime.fromtimestamp(os.path.getctime(filename))
    expired = created + self.duration
    if expired < datetime.datetime.now():
        os.remove(filename)
        log.debug('%s expired, deleted', filename)