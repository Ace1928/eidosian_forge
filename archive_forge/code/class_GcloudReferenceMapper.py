from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
import subprocess
import sys
import webbrowser
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
class GcloudReferenceMapper(CommandReferenceMapper):
    """gcloud help reference mapper."""

    def GetMan(self):
        return ' '.join(self.args + ['--help'])

    def GetURL(self):
        return '/'.join(['https://cloud.google.com/sdk/gcloud/reference'] + self.args[1:])