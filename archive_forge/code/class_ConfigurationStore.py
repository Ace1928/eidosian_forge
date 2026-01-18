from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import threading
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import properties_file
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
class ConfigurationStore(object):
    """Class for performing low level operations on configs and their files."""

    @staticmethod
    def ActiveConfig():
        """Gets the currently active configuration.

    There must always be an active configuration.  If there isn't this means
    no configurations have been created yet and this will auto-create a default
    configuration.  If there are legacy user properties, they will be migrated
    to the newly created configuration.

    Returns:
      Configuration, the currently active configuration.
    """
        return ActiveConfig(force_create=True)

    @staticmethod
    def AllConfigs(include_none_config=False):
        """Returns all the configurations that exist.

    This determines the currently active configuration so as a side effect it
    will create the default configuration if no configurations exist.

    Args:
      include_none_config: bool, True to include the NONE configuration in the
        list. This is a reserved configuration that indicates to not use any
        configuration.  It is not explicitly created but is always available.

    Returns:
      {str, Configuration}, A map of configuration name to the configuration
      object.
    """
        config_dir = config.Paths().named_config_directory
        active_config = ConfigurationStore.ActiveConfig()
        active_config_name = active_config.name
        configs = {}
        if include_none_config:
            configs[_NO_ACTIVE_CONFIG_NAME] = Configuration(_NO_ACTIVE_CONFIG_NAME, _NO_ACTIVE_CONFIG_NAME == active_config_name)
        try:
            config_files = os.listdir(config_dir)
            for f in config_files:
                m = re.match(_CONFIG_FILE_REGEX, f)
                if m:
                    name = m.group(1)
                    configs[name] = Configuration(name, name == active_config_name)
            return configs
        except (OSError, IOError) as exc:
            if exc.errno != errno.ENOENT:
                raise NamedConfigFileAccessError('List of configurations could not be read from: [{0}]'.format(config_dir), exc)
        return {}

    @staticmethod
    def CreateConfig(config_name):
        """Creates a configuration with the given name.

    Args:
      config_name: str, The name of the configuration to create.

    Returns:
      Configuration, The configuration that was just created.

    Raises:
      NamedConfigError: If the configuration already exists.
      NamedConfigFileAccessError: If there a problem manipulating the
        configuration files.
    """
        _EnsureValidConfigName(config_name, allow_reserved=False)
        paths = config.Paths()
        file_path = _FileForConfig(config_name, paths)
        if os.path.exists(file_path):
            raise NamedConfigError('Cannot create configuration [{0}], it already exists.'.format(config_name))
        try:
            file_utils.MakeDir(paths.named_config_directory)
            file_utils.WriteFileContents(file_path, '')
        except file_utils.Error as e:
            raise NamedConfigFileAccessError('Failed to create configuration [{0}].  Ensure you have the correct permissions on [{1}]'.format(config_name, paths.named_config_directory), e)
        return Configuration(config_name, is_active=False)

    @staticmethod
    def DeleteConfig(config_name):
        """Creates the given configuration.

    Args:
      config_name: str, The name of the configuration to delete.

    Raises:
      NamedConfigError: If the configuration does not exist.
      NamedConfigFileAccessError: If there a problem manipulating the
        configuration files.
    """
        _EnsureValidConfigName(config_name, allow_reserved=False)
        paths = config.Paths()
        file_path = _FileForConfig(config_name, paths)
        if not os.path.exists(file_path):
            raise NamedConfigError('Cannot delete configuration [{0}], it does not exist.'.format(config_name))
        if config_name == _EffectiveActiveConfigName():
            raise NamedConfigError('Cannot delete configuration [{0}], it is the currently active configuration.'.format(config_name))
        if config_name == _ActiveConfigNameFromFile():
            raise NamedConfigError('Cannot delete configuration [{0}], it is currently set as the active configuration in your gcloud properties.'.format(config_name))
        try:
            os.remove(file_path)
        except (OSError, IOError) as e:
            raise NamedConfigFileAccessError('Failed to delete configuration [{0}].  Ensure you have the correct permissions on [{1}]'.format(config_name, paths.named_config_directory), e)

    @staticmethod
    def ActivateConfig(config_name):
        """Activates an existing named configuration.

    Args:
      config_name: str, The name of the configuration to activate.

    Raises:
      NamedConfigError: If the configuration does not exists.
      NamedConfigFileAccessError: If there a problem manipulating the
        configuration files.
    """
        _EnsureValidConfigName(config_name, allow_reserved=True)
        paths = config.Paths()
        file_path = _FileForConfig(config_name, paths)
        if file_path and (not os.path.exists(file_path)):
            raise NamedConfigError('Cannot activate configuration [{0}], it does not exist.'.format(config_name))
        try:
            file_utils.WriteFileContents(paths.named_config_activator_path, config_name)
        except file_utils.Error as e:
            raise NamedConfigFileAccessError('Failed to activate configuration [{0}].  Ensure you have the correct permissions on [{1}]'.format(config_name, paths.named_config_activator_path), e)
        ActivePropertiesFile.Invalidate(mark_changed=True)

    @staticmethod
    def RenameConfig(config_name, new_name):
        """Renames an existing named configuration.

    Args:
      config_name: str, The name of the configuration to rename.
      new_name: str, The new name of the configuration.

    Raises:
      NamedConfigError: If the configuration does not exist, or if the
        configuration with new_name exists.
      NamedConfigFileAccessError: If there a problem manipulating the
        configuration files.
    """
        _EnsureValidConfigName(new_name, allow_reserved=True)
        paths = config.Paths()
        file_path = _FileForConfig(config_name, paths)
        if file_path and (not os.path.exists(file_path)):
            raise NamedConfigError('Cannot rename configuration [{0}], it does not exist.'.format(config_name))
        if config_name == _EffectiveActiveConfigName():
            raise NamedConfigError('Cannot rename configuration [{0}], it is the currently active configuration.'.format(config_name))
        if config_name == _ActiveConfigNameFromFile():
            raise NamedConfigError('Cannot rename configuration [{0}], it is currently set as the active configuration in your gcloud properties.'.format(config_name))
        new_file_path = _FileForConfig(new_name, paths)
        if new_file_path and os.path.exists(new_file_path):
            raise NamedConfigError('Cannot rename configuration [{0}], [{1}] already exists.'.format(config_name, new_name))
        try:
            contents = file_utils.ReadFileContents(file_path)
            file_utils.WriteFileContents(new_file_path, contents)
            os.remove(file_path)
        except file_utils.Error as e:
            raise NamedConfigFileAccessError('Failed to rename configuration [{0}].  Ensure you have the correct permissions on [{1}]'.format(config_name, paths.named_config_activator_path), e)