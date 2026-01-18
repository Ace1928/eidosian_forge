from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
from castellan import key_manager
from oslo_config import cfg
from oslo_config import sources
from oslo_log import log
class CastellanConfigurationSource(sources.ConfigurationSource):
    """A configuration source for configuration values served through castellan.  # noqa

    :param config_file: The path to a castellan configuration file.

    :param mapping_file: The path to a configuration/castellan_id mapping file.
    """

    def __init__(self, group_name, config_file, mapping_file):
        conf = cfg.ConfigOpts()
        conf(args=[], default_config_files=[config_file])
        self._name = group_name
        self._mngr = key_manager.API(conf)
        self._mapping = {}
        cfg.ConfigParser(mapping_file, self._mapping).parse()

    def get(self, group_name, option_name, opt):
        try:
            group_name = group_name or 'DEFAULT'
            castellan_id = self._mapping[group_name][option_name][0]
            return (self._mngr.get('ctx', castellan_id).get_encoded().decode(), cfg.LocationInfo(cfg.Locations.user, castellan_id))
        except KeyError:
            LOG.info("option '[%s] %s' not present in '[%s] mapping_file'", group_name, option_name, self._name)
        except KeyManagerError:
            LOG.warning("missing castellan_id for option '[%s] %s' in '[%s] mapping_file'", group_name, option_name, self._name)
        except ManagedObjectNotFoundError:
            LOG.warning("invalid castellan_id for option '[%s] %s' in '[%s] mapping_file'", group_name, option_name, self._name)
        return (sources._NoValue, None)