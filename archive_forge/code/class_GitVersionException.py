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
class GitVersionException(Error):
    """Exceptions for when git version is too old."""

    def __init__(self, fmtstr, cur_version, min_version):
        self.cur_version = cur_version
        super(GitVersionException, self).__init__(fmtstr.format(cur_version=cur_version, min_version=min_version))