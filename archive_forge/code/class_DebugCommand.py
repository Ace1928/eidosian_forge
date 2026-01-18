import importlib.resources
import locale
import logging
import os
import sys
from optparse import Values
from types import ModuleType
from typing import Any, Dict, List, Optional
import pip._vendor
from pip._vendor.certifi import where
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.configuration import Configuration
from pip._internal.metadata import get_environment
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_pip_version
class DebugCommand(Command):
    """
    Display debug information.
    """
    usage = '\n      %prog <options>'
    ignore_require_venv = True

    def add_options(self) -> None:
        cmdoptions.add_target_python_options(self.cmd_opts)
        self.parser.insert_option_group(0, self.cmd_opts)
        self.parser.config.load()

    def run(self, options: Values, args: List[str]) -> int:
        logger.warning('This command is only meant for debugging. Do not use this with automation for parsing and getting these details, since the output and options of this command may change without notice.')
        show_value('pip version', get_pip_version())
        show_value('sys.version', sys.version)
        show_value('sys.executable', sys.executable)
        show_value('sys.getdefaultencoding', sys.getdefaultencoding())
        show_value('sys.getfilesystemencoding', sys.getfilesystemencoding())
        show_value('locale.getpreferredencoding', locale.getpreferredencoding())
        show_value('sys.platform', sys.platform)
        show_sys_implementation()
        show_value("'cert' config value", ca_bundle_info(self.parser.config))
        show_value('REQUESTS_CA_BUNDLE', os.environ.get('REQUESTS_CA_BUNDLE'))
        show_value('CURL_CA_BUNDLE', os.environ.get('CURL_CA_BUNDLE'))
        show_value('pip._vendor.certifi.where()', where())
        show_value('pip._vendor.DEBUNDLED', pip._vendor.DEBUNDLED)
        show_vendor_versions()
        show_tags(options)
        return SUCCESS