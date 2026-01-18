from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import appinfo_includes
from googlecloudsdk.third_party.appengine.api import croninfo
from googlecloudsdk.third_party.appengine.api import dispatchinfo
from googlecloudsdk.third_party.appengine.api import queueinfo
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.datastore import datastore_index
class ConfigYamlInfo(_YamlInfo):
    """A class for holding some basic attributes of a parsed config .yaml file."""
    CRON = 'cron'
    DISPATCH = 'dispatch'
    INDEX = 'index'
    QUEUE = 'queue'
    CONFIG_YAML_PARSERS = {CRON: croninfo.LoadSingleCron, DISPATCH: dispatchinfo.LoadSingleDispatch, INDEX: datastore_index.ParseIndexDefinitions, QUEUE: queueinfo.LoadSingleQueue}

    def __init__(self, file_path, config, parsed):
        """Creates a new ConfigYamlInfo.

    Args:
      file_path: str, The full path the file that was parsed.
      config: str, The name of the config that was parsed (i.e. 'cron')
      parsed: The parsed yaml data as one of the *_info objects.
    """
        super(ConfigYamlInfo, self).__init__(file_path, parsed)
        self.config = config

    @property
    def name(self):
        """Name of the config file without extension, e.g. `cron`."""
        base, _ = os.path.splitext(os.path.basename(self.file))
        return base

    @staticmethod
    def FromFile(file_path):
        """Parses the given config file.

    Args:
      file_path: str, The full path to the config file.

    Raises:
      Error: If a user tries to parse a dos.yaml file.
      YamlParseError: If the file is not valid.

    Returns:
      A ConfigYamlInfo object for the parsed file.
    """
        base, ext = os.path.splitext(os.path.basename(file_path))
        if base == 'dos':
            raise Error('`gcloud app deploy dos.yaml` is no longer supported. Please use `gcloud app firewall-rules` instead.')
        parser = ConfigYamlInfo.CONFIG_YAML_PARSERS.get(base) if os.path.isfile(file_path) and ext.lower() in ['.yaml', '.yml'] else None
        if not parser:
            return None
        try:
            parsed = _YamlInfo._ParseYaml(file_path, parser)
            if not parsed:
                raise YamlParseError(file_path, 'The file is empty')
        except (yaml_errors.Error, validation.Error) as e:
            raise YamlParseError(file_path, e)
        _CheckIllegalAttribute(name='application', yaml_info=parsed, extractor_func=lambda yaml: yaml.application, file_path=file_path, msg=HINT_PROJECT)
        if base == 'dispatch':
            return DispatchConfigYamlInfo(file_path, config=base, parsed=parsed)
        return ConfigYamlInfo(file_path, config=base, parsed=parsed)