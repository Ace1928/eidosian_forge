from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
import_site_packages = (os.environ.get(SITE_PACKAGES) or
from googlecloudsdk.core.util import platforms
def DoAllRequiredChecks():
    if not platforms.PythonVersion().IsCompatible():
        sys.exit(1)