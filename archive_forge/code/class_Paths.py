from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
class Paths(object):
    """Class to encapsulate the various directory paths of the Cloud SDK.

  Attributes:
    global_config_dir: str, The path to the user's global config area.
  """
    CLOUDSDK_STATE_DIR = '.install'
    CLOUDSDK_PROPERTIES_NAME = 'properties'
    _global_config_dir = None

    def __init__(self):
        self._global_config_dir = _GetGlobalConfigDir()

    @property
    def global_config_dir(self):
        return self._global_config_dir

    @property
    def sdk_root(self):
        """Searches for the Cloud SDK root directory.

    Returns:
      str, The path to the root of the Cloud SDK or None if it could not be
      found.
    """
        return file_utils.FindDirectoryContaining(os.path.dirname(encoding.Decode(__file__)), Paths.CLOUDSDK_STATE_DIR)

    @property
    def sdk_bin_path(self):
        """Forms a path to bin directory by using sdk_root.

    Returns:
      str, The path to the bin directory of the Cloud SDK or None if it could
      not be found.
    """
        sdk_root = self.sdk_root
        return os.path.join(sdk_root, 'bin') if sdk_root else None

    @property
    def cache_dir(self):
        """Gets the dir path that will contain all cache objects."""
        return os.path.join(self.global_config_dir, 'cache')

    @property
    def credentials_db_path(self):
        """Gets the path to the file to store credentials in.

    This is generic key/value store format using sqlite.

    Returns:
      str, The path to the credential db file.
    """
        return os.path.join(self.global_config_dir, 'credentials.db')

    @property
    def config_db_path(self):
        """Gets the path to the file to store configs in.

    This is generic key/value store format using sqlite.

    Returns:
      str, The path to the config db file.
    """
        return os.path.join(self.global_config_dir, '{}_configs.db')

    @property
    def access_token_db_path(self):
        """Gets the path to the file to store cached access tokens in.

    This is generic key/value store format using sqlite.

    Returns:
      str, The path to the access token db file.
    """
        return os.path.join(self.global_config_dir, 'access_tokens.db')

    @property
    def logs_dir(self):
        """Gets the path to the directory to put logs in for calliope commands.

    Returns:
      str, The path to the directory to put logs in.
    """
        return os.path.join(self.global_config_dir, 'logs')

    @property
    def cid_path(self):
        """Gets the path to the file to store the client ID.

    Returns:
      str, The path to the file.
    """
        return os.path.join(self.global_config_dir, '.metricsUUID')

    @property
    def feature_flags_config_path(self):
        """Gets the path to the file to store the cached feature flags config file.

    Returns:
      str, The path to the file.
    """
        return os.path.join(self.global_config_dir, '.feature_flags_config.yaml')

    @property
    def update_check_cache_path(self):
        """Gets the path to the file to cache information about update checks.

    This is stored in the config directory instead of the installation state
    because if the SDK is installed as root, it will fail to persist the cache
    when you are running gcloud as a normal user.

    Returns:
      str, The path to the file.
    """
        return os.path.join(self.global_config_dir, '.last_update_check.json')

    @property
    def survey_prompting_cache_path(self):
        """Gets the path to the file to cache information about survey prompting.

    This is stored in the config directory instead of the installation state
    because if the SDK is installed as root, it will fail to persist the cache
    when you are running gcloud as a normal user.

    Returns:
      str, The path to the file.
    """
        return os.path.join(self.global_config_dir, '.last_survey_prompt.yaml')

    @property
    def opt_in_prompting_cache_path(self):
        """Gets the path to the file to cache information about opt-in prompting.

    This is stored in the config directory instead of the installation state
    because if the SDK is installed as root, it will fail to persist the cache
    when you are running gcloud as a normal user.

    Returns:
      str, The path to the file.
    """
        return os.path.join(self.global_config_dir, '.last_opt_in_prompt.yaml')

    @property
    def installation_properties_path(self):
        """Gets the path to the installation-wide properties file.

    Returns:
      str, The path to the file.
    """
        sdk_root = self.sdk_root
        if not sdk_root:
            return None
        return os.path.join(sdk_root, self.CLOUDSDK_PROPERTIES_NAME)

    @property
    def user_properties_path(self):
        """Gets the path to the properties file in the user's global config dir.

    Returns:
      str, The path to the file.
    """
        return os.path.join(self.global_config_dir, self.CLOUDSDK_PROPERTIES_NAME)

    @property
    def named_config_activator_path(self):
        """Gets the path to the file pointing at the user's active named config.

    This is the file that stores the name of the user's active named config,
    not the path to the configuration file itself.

    Returns:
      str, The path to the file.
    """
        return os.path.join(self.global_config_dir, 'active_config')

    @property
    def named_config_directory(self):
        """Gets the path to the directory that stores the named configs.

    Returns:
      str, The path to the directory.
    """
        return os.path.join(self.global_config_dir, 'configurations')

    @property
    def config_sentinel_file(self):
        """Gets the path to the config sentinel.

    The sentinel is a file that we touch any time there is a change to config.
    External tools can check this file to see if they need to re-query gcloud's
    credential/config helper to get updated configuration information. Nothing
    is ever written to this file, it's timestamp indicates the last time config
    was changed.

    This does not take into account config changes made through environment
    variables as they are transient by nature. There is also the edge case of
    when a user updated installation config. That user's sentinel will be
    updated but other will not be.

    Returns:
      str, The path to the sentinel file.
    """
        return os.path.join(self.global_config_dir, 'config_sentinel')

    @property
    def valid_ppk_sentinel_file(self):
        """Gets the path to the sentinel used to check for PPK encoding validity.

    The presence of this file is simply used to indicate whether or not we've
    correctly encoded the PPK used for ssh on Windows (re-encoding may be
    necessary in order to fix a bug in an older version of winkeygen.exe).

    Returns:
      str, The path to the sentinel file.
    """
        return os.path.join(self.global_config_dir, '.valid_ppk_sentinel')

    @property
    def container_config_path(self):
        """Absolute path of the container config dir."""
        return os.path.join(self.global_config_dir, 'kubernetes')

    @property
    def virtualenv_dir(self):
        """Absolute path of the virtual env dir."""
        return os.path.join(self.global_config_dir, 'virtenv')

    def LegacyCredentialsDir(self, account):
        """Gets the path to store legacy credentials in.

    Args:
      account: str, Email account tied to the authorizing credentials.

    Returns:
      str, The path to the credentials file.
    """
        if not account:
            account = 'default'
        account = account.replace(':', '')
        if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS and (account.upper().startswith('CON.') or account.upper().startswith('PRN.') or account.upper().startswith('AUX.') or account.upper().startswith('NUL.')):
            account = '.' + account
        return os.path.join(self.global_config_dir, 'legacy_credentials', account)

    def LegacyCredentialsBqPath(self, account):
        """Gets the path to store legacy bq credentials in.

    Args:
      account: str, Email account tied to the authorizing credentials.

    Returns:
      str, The path to the bq credentials file.
    """
        return os.path.join(self.LegacyCredentialsDir(account), 'singlestore_bq.json')

    def LegacyCredentialsGSUtilPath(self, account):
        """Gets the path to store legacy gsutil credentials in.

    Args:
      account: str, Email account tied to the authorizing credentials.

    Returns:
      str, The path to the gsutil credentials file.
    """
        return os.path.join(self.LegacyCredentialsDir(account), '.boto')

    def LegacyCredentialsP12KeyPath(self, account):
        """Gets the path to store legacy key file in.

    Args:
      account: str, Email account tied to the authorizing credentials.

    Returns:
      str, The path to the key file.
    """
        return os.path.join(self.LegacyCredentialsDir(account), 'private_key.p12')

    def LegacyCredentialsAdcPath(self, account):
        """Gets the file path to store application default credentials in.

    Args:
      account: str, Email account tied to the authorizing credentials.

    Returns:
      str, The path to the file.
    """
        return os.path.join(self.LegacyCredentialsDir(account), 'adc.json')

    def GCECachePath(self):
        """Get the path to cache whether or not we're on a GCE machine.

    Returns:
      str, The path to the GCE cache.
    """
        return os.path.join(self.global_config_dir, 'gce')