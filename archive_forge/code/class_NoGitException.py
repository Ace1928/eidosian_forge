from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import subprocess
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
import uritemplate
class NoGitException(Error):
    """Exceptions for when git is not available."""

    def __init__(self):
        super(NoGitException, self).__init__(textwrap.dedent('        Cannot find git. Please install git and try again.\n\n        You can find git installers at [http://git-scm.com/downloads], or use\n        your favorite package manager to install it on your computer. Make sure\n        it can be found on your system PATH.\n        '))