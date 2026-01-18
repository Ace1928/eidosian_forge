import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def _delete_temp(self, fd, name):
    fd.close()
    os.unlink(name)