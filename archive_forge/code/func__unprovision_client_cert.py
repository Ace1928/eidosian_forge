from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
from boto import config
import gslib
from gslib import exception
from gslib.utils import boto_util
from gslib.utils import execution_util
def _unprovision_client_cert(self):
    """Cleans up any files or resources provisioned during config init."""
    if self.client_cert_path is not None:
        try:
            os.remove(self.client_cert_path)
            self.logger.debug('Unprovisioned client cert: %s' % self.client_cert_path)
        except OSError as e:
            self.logger.error('Failed to remove client certificate: %s' % e)