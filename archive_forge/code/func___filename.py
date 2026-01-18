import suds
import suds.sax.element
import suds.sax.parser
import datetime
import os
import shutil
import tempfile
from logging import getLogger
def __filename(self, id):
    """Return the cache file name for an entry with a given id."""
    suffix = self.fnsuffix()
    filename = '%s-%s.%s' % (self.fnprefix, id, suffix)
    return os.path.join(self.location, filename)