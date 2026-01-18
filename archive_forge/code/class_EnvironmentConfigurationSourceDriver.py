import os
import oslo_config.cfg
from oslo_config import sources
class EnvironmentConfigurationSourceDriver(sources.ConfigurationSourceDriver):
    """A backend driver for environment variables.

    This configuration source is available by default and does not need special
    configuration to use. The sample config is generated automatically but is
    not necessary.
    """

    def list_options_for_discovery(self):
        """There are no options for this driver."""
        return []

    def open_source_from_opt_group(self, conf, group_name):
        return EnvironmentConfigurationSource()