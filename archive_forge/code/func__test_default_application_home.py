import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def _test_default_application_home(self):
    """
        application home

        """
    app_home = self.ETSConfig.application_home
    dirname, app_name = os.path.split(app_home)
    self.assertEqual(dirname, self.ETSConfig.application_data)
    self.assertEqual(app_name, 'tests')