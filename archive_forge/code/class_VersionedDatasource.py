import configparser
import glob
import os
import sys
from os.path import join as pjoin
from packaging.version import Version
from .environment import get_nipy_system_dir, get_nipy_user_dir
class VersionedDatasource(Datasource):
    """Datasource with version information in config file"""

    def __init__(self, base_path, config_filename=None):
        """Initialize versioned datasource

        We assume that there is a configuration file with version
        information in datasource directory tree.

        The configuration file contains an entry like::

           [DEFAULT]
           version = 0.3

        The version should have at least a major and a minor version
        number in the form above.

        Parameters
        ----------
        base_path : str
           path to prepend to all relative paths
        config_filaname : None or str
           relative path to configuration file containing version

        """
        Datasource.__init__(self, base_path)
        if config_filename is None:
            config_filename = 'config.ini'
        self.config = configparser.ConfigParser()
        cfg_file = self.get_filename(config_filename)
        readfiles = self.config.read(cfg_file)
        if not readfiles:
            raise DataError(f'Could not read config file {cfg_file}')
        try:
            self.version = self.config.get('DEFAULT', 'version')
        except configparser.Error:
            raise DataError(f'Could not get version from {cfg_file}')
        version_parts = self.version.split('.')
        self.major_version = int(version_parts[0])
        self.minor_version = int(version_parts[1])
        self.version_no = float(f'{self.major_version}.{self.minor_version}')