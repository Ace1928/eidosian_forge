from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
from castellan import key_manager
from oslo_config import cfg
from oslo_config import sources
from oslo_log import log
class CastellanConfigurationSourceDriver(sources.ConfigurationSourceDriver):
    """A backend driver for configuration values served through castellan.

    Required options:
      - config_file: The castellan configuration file.

      - mapping_file: A configuration/castellan_id mapping file. This file
                      creates connections between configuration options and
                      castellan ids. The group and option name remains the
                      same, while the value gets stored a secret manager behind
                      castellan and is replaced by its castellan id. The ids
                      will be used to fetch the values through castellan.
    """
    _castellan_driver_opts = [cfg.StrOpt('config_file', required=True, sample_default='etc/castellan/castellan.conf', help='The path to a castellan configuration file.'), cfg.StrOpt('mapping_file', required=True, sample_default='etc/castellan/secrets_mapping.conf', help='The path to a configuration/castellan_id mapping file.')]

    def list_options_for_discovery(self):
        return self._castellan_driver_opts

    def open_source_from_opt_group(self, conf, group_name):
        conf.register_opts(self._castellan_driver_opts, group_name)
        return CastellanConfigurationSource(group_name, conf[group_name].config_file, conf[group_name].mapping_file)