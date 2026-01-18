from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import bootstrapping
import argparse
import os
import sys
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import platforms_install
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk import gcloud_main
def MaybeInstallPythonOnMac():
    """Optionally install Python on Mac machines."""
    if platforms.OperatingSystem.Current() != platforms.OperatingSystem.MACOSX:
        return
    print('\nGoogle Cloud CLI works best with Python {} and certain modules.\n'.format(PYTHON_VERSION))
    already_have_python_version = os.path.isdir(MACOS_PYTHON_INSTALL_PATH)
    if already_have_python_version:
        prompt = 'Python {} installation detected, install recommended modules?'.format(PYTHON_VERSION)
    else:
        prompt = 'Download and run Python {} installer?'.format(PYTHON_VERSION)
    setup_python = console_io.PromptContinue(prompt_string=prompt, default=True)
    if setup_python:
        install_errors = []
        if not already_have_python_version:
            print('Running Python {} installer, you may be prompted for sudo password...'.format(PYTHON_VERSION))
            with files.TemporaryDirectory() as tempdir:
                with files.ChDir(tempdir):
                    curl_args = ['curl', '--silent', '-O', MACOS_PYTHON_URL]
                    exit_code = execution_utils.Exec(curl_args, no_exit=True)
                    if exit_code != 0:
                        install_errors.append('Failed to download Python installer')
                    else:
                        exit_code = execution_utils.Exec(['tar', '-xf', MACOS_PYTHON], no_exit=True)
                        if exit_code != 0:
                            install_errors.append('Failed to extract Python installer')
                        else:
                            exit_code = execution_utils.Exec(['sudo', 'installer', '-target', '/', '-pkg', './python-3.11.6-macos11.pkg'], no_exit=True)
                            if exit_code != 0:
                                install_errors.append('Installer failed.')
        if not install_errors:
            python_to_use = '{}/bin/python3'.format(MACOS_PYTHON_INSTALL_PATH)
            os.environ['CLOUDSDK_PYTHON'] = python_to_use
            print('Setting up virtual environment')
            if os.path.isdir(config.Paths().virtualenv_dir):
                _CLI.Execute(['config', 'virtualenv', 'update'])
                _CLI.Execute(['config', 'virtualenv', 'enable'])
            else:
                _CLI.Execute(['config', 'virtualenv', 'create', '--python-to-use', python_to_use])
                _CLI.Execute(['config', 'virtualenv', 'enable'])
        else:
            print('Failed to install Python. Errors \n\n{}'.format('\n*'.join(install_errors)))