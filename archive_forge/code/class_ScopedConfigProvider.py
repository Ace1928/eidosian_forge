import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class ScopedConfigProvider(BaseProvider):

    def __init__(self, config_var_name, session):
        """Initialize ScopedConfigProvider.

        :type config_var_name: str or tuple
        :param config_var_name: The name of the config variable to load from
            the configuration file. If the value is a tuple, it must only
            consist of two items, where the first item represents the section
            and the second item represents the config var name in the section.

        :type session: :class:`botocore.session.Session`
        :param session: The botocore session to get the loaded configuration
            file variables from.
        """
        self._config_var_name = config_var_name
        self._session = session

    def __deepcopy__(self, memo):
        return ScopedConfigProvider(copy.deepcopy(self._config_var_name, memo), self._session)

    def provide(self):
        """Provide a value from a config file property."""
        scoped_config = self._session.get_scoped_config()
        if isinstance(self._config_var_name, tuple):
            section_config = scoped_config.get(self._config_var_name[0])
            if not isinstance(section_config, dict):
                return None
            return section_config.get(self._config_var_name[1])
        return scoped_config.get(self._config_var_name)

    def __repr__(self):
        return 'ScopedConfigProvider(config_var_name={}, session={})'.format(self._config_var_name, self._session)