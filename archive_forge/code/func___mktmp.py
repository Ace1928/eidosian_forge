import suds
import suds.sax.element
import suds.sax.parser
import datetime
import os
import shutil
import tempfile
from logging import getLogger
def __mktmp(self):
    """Create the I{location} folder if it does not already exist."""
    try:
        if not os.path.isdir(self.location):
            os.makedirs(self.location)
    except Exception:
        log.debug(self.location, exc_info=1)
    return self