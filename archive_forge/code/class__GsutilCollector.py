from a man-ish style runtime document.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.static_completion import generate as generate_static
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import range
class _GsutilCollector(object):
    """gsutil help document section collector."""
    UNKNOWN, ROOT, MAN, TOPIC = list(range(4))

    def __init__(self, text):
        self.text = text.split('\n')
        self.heading = 'CAPSULE'
        self.page_type = self.UNKNOWN

    def Collect(self, strip_headings=False):
        """Returns the heading and content lines from text."""
        content = []
        heading = self.heading
        self.heading = None
        while self.text:
            line = self.text.pop(0)
            if self.page_type == self.UNKNOWN:
                if line.startswith('Usage:'):
                    self.page_type = self.ROOT
                    continue
                elif line == 'NAME':
                    self.page_type = self.MAN
                    heading = 'CAPSULE'
                    continue
                elif not line.startswith(' '):
                    continue
            elif self.page_type == self.ROOT:
                if line == 'Available commands:':
                    heading = 'COMMANDS'
                    continue
                elif line == 'Additional help topics:':
                    self.heading = 'TOPICS'
                    break
                elif not line.startswith(' '):
                    continue
            elif self.page_type == self.MAN:
                if line == 'OVERVIEW':
                    self.page_type = self.TOPIC
                    self.heading = 'DESCRIPTION'
                    break
                elif line == 'SYNOPSIS':
                    self.heading = line
                    break
                elif line.endswith('OPTIONS'):
                    self.heading = 'FLAGS'
                    break
                elif line and line[0].isupper():
                    self.heading = line.split(' ', 1)[-1]
                    break
            elif self.page_type == self.TOPIC:
                if line and line[0].isupper():
                    self.heading = line
                    break
            if line.startswith(' ') or not strip_headings:
                content.append(line.rstrip())
        while content and (not content[0]):
            content.pop(0)
        while content and (not content[-1]):
            content.pop()
        return (heading, content)