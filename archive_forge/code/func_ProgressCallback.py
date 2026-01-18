from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
import sys
from apitools.base.py import transfer
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import transports
def ProgressCallback(_, download):
    """callback function to print the progress of the download."""
    if download.total_size:
        progress = download.progress / download.total_size
        if progress < 1:
            progress_bar.SetProgress(progress)