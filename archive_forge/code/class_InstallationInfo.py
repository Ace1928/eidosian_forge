from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import getpass
import io
import locale
import os
import platform as system_platform
import re
import ssl
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import http_proxy_setup
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import requests
import six
import urllib3
class InstallationInfo(object):
    """Holds information about your Cloud SDK installation."""

    def __init__(self, anonymizer=None):
        anonymizer = anonymizer or NoopAnonymizer()
        self.sdk_root = anonymizer.ProcessPath(config.Paths().sdk_root)
        self.release_channel = config.INSTALLATION_CONFIG.release_channel
        self.repo_url = anonymizer.ProcessURL(config.INSTALLATION_CONFIG.snapshot_url)
        repos = properties.VALUES.component_manager.additional_repositories.Get(validate=False)
        self.additional_repos = map(anonymizer.ProcessURL, repos.split(',')) if repos else []
        path = encoding.GetEncodedValue(os.environ, 'PATH', '').split(os.pathsep)
        self.python_path = [anonymizer.ProcessPath(encoding.Decode(path_elem)) for path_elem in sys.path]
        if self.sdk_root:
            manager = update_manager.UpdateManager()
            self.components = manager.GetCurrentVersionsInformation()
            self.other_tool_paths = [anonymizer.ProcessPath(p) for p in manager.FindAllOtherToolsOnPath()]
            self.duplicate_tool_paths = [anonymizer.ProcessPath(p) for p in manager.FindAllDuplicateToolsOnPath()]
            paths = [os.path.realpath(p) for p in path]
            this_path = os.path.realpath(os.path.join(self.sdk_root, update_manager.UpdateManager.BIN_DIR_NAME))
            self.on_path = this_path in paths
        else:
            self.components = {}
            self.other_tool_paths = []
            self.duplicate_tool_paths = []
            self.on_path = False
        self.path = [anonymizer.ProcessPath(p) for p in path]
        self.kubectl = file_utils.SearchForExecutableOnPath('kubectl')
        if self.kubectl:
            self.kubectl = anonymizer.ProcessPath(self.kubectl[0])

    def __str__(self):
        out = io.StringIO()
        out.write('Installation Root: [{0}]\n'.format(self.sdk_root if self.sdk_root else 'N/A'))
        if config.INSTALLATION_CONFIG.IsAlternateReleaseChannel():
            out.write('Release Channel: [{0}]\n'.format(self.release_channel))
            out.write('Repository URL: [{0}]\n'.format(self.repo_url))
        if self.additional_repos:
            out.write('Additional Repositories:\n  {0}\n'.format('\n  '.join(self.additional_repos)))
        if self.components:
            components = ['{0}: [{1}]'.format(name, value) for name, value in six.iteritems(self.components)]
            out.write('Installed Components:\n  {0}\n'.format('\n  '.join(components)))
        out.write('System PATH: [{0}]\n'.format(os.pathsep.join(self.path)))
        out.write('Python PATH: [{0}]\n'.format(os.pathsep.join(self.python_path)))
        out.write('Cloud SDK on PATH: [{0}]\n'.format(self.on_path))
        out.write('Kubectl on PATH: [{0}]\n'.format(self.kubectl or False))
        if self.other_tool_paths:
            out.write('\nWARNING: There are other instances of the Google Cloud Platform tools on your system PATH.\n  {0}\n'.format('\n  '.join(self.other_tool_paths)))
        if self.duplicate_tool_paths:
            out.write('There are alternate versions of the following Google Cloud Platform tools on your system PATH.\n  {0}\n'.format('\n  '.join(self.duplicate_tool_paths)))
        return out.getvalue()