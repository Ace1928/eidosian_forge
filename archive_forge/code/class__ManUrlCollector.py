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
class _ManUrlCollector(_ManPageCollector):
    """man URL help document section collector."""
    _CLI_VERSION = 'man7.org-0.1'

    @classmethod
    def GetVersion(cls):
        return cls._CLI_VERSION

    def _GetRawManPageText(self):
        """Returns the raw man page text."""
        session = requests.GetSession()
        url = 'http://man7.org/linux/man-pages/man1/{}.1.html'.format(self.command_name)
        response = session.get(url)
        if response.status_code != 200:
            raise NoManPageTextForCommandError('Cannot get URL man page text for [{}].'.format(self.command_name))
        return response.text

    def GetManPageText(self):
        """Returns the text man page for self.command_name from a URL."""
        text = self._GetRawManPageText()
        for pattern, replacement in (('<span class="footline">.*', ''), ('<h2><a id="([^"]*)"[^\n]*\n', '\\1\n'), ('<b>( +)', '\\1*'), ('( +)</b>', '*\\1'), ('<i>( +)', '\\1_'), ('( +)</i>', '_\\1'), ('</?b>', '*'), ('</?i>', '_'), ('</pre>', ''), ('<a href="([^"]*)">([^\n]*)</a>', '[\\1](\\2)'), ('&amp;', '\\&'), ('&gt;', '>'), ('&lt;', '<'), ('&#39;', "'")):
            text = re.sub(pattern, replacement, text, flags=re.DOTALL)
        lines = []
        top = 'NAME'
        flags = False
        paragraph = False
        for line in text.split('\n'):
            if top and line == 'NAME':
                top = None
                lines = []
            if line.startswith('       *-'):
                flags = True
                if paragraph:
                    paragraph = False
                    lines.append('')
                if '  ' in line[7:]:
                    head, tail = line[7:].split('  ', 1)
                    head = re.sub('\\*', '', head)
                    line = '  '.join(['     ', head, tail])
                else:
                    line = re.sub('\\*', '', line)
            elif flags:
                if not line:
                    paragraph = True
                    continue
                elif not line.startswith('       '):
                    flags = False
                    paragraph = False
                elif paragraph:
                    if not line[0].lower():
                        lines.append('+')
                    paragraph = False
            lines.append(line)
        return '\n'.join(lines)