import argparse
import contextlib
import os
import sys
from configparser import ConfigParser
from typing import Dict, List, Union
from docformatter import __pkginfo__
def _do_read_configuration_file(self) -> None:
    """Read docformatter options from a configuration file."""
    argfile = os.path.basename(self.config_file)
    for f in self.configuration_file_lst:
        if argfile == f:
            break
    fullpath, ext = os.path.splitext(self.config_file)
    filename = os.path.basename(fullpath)
    if ext == '.toml' and (TOMLI_INSTALLED or TOMLLIB_INSTALLED) and (filename == 'pyproject'):
        self._do_read_toml_configuration()
    if ext == '.cfg' and filename == 'setup' or (ext == '.ini' and filename == 'tox'):
        self._do_read_parser_configuration()