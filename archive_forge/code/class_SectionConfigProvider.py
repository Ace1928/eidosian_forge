import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class SectionConfigProvider(BaseProvider):
    """Provides a dictionary from a section in the scoped config

    This is useful for retrieving scoped config variables (i.e. s3) that have
    their own set of config variables and resolving logic.
    """

    def __init__(self, section_name, session, override_providers=None):
        self._section_name = section_name
        self._session = session
        self._scoped_config_provider = ScopedConfigProvider(self._section_name, self._session)
        self._override_providers = override_providers
        if self._override_providers is None:
            self._override_providers = {}

    def __deepcopy__(self, memo):
        return SectionConfigProvider(copy.deepcopy(self._section_name, memo), self._session, copy.deepcopy(self._override_providers, memo))

    def provide(self):
        section_config = self._scoped_config_provider.provide()
        if section_config and (not isinstance(section_config, dict)):
            logger.debug('The %s config key is not a dictionary type, ignoring its value of: %s', self._section_name, section_config)
            return None
        for section_config_var, provider in self._override_providers.items():
            provider_val = provider.provide()
            if provider_val is not None:
                if section_config is None:
                    section_config = {}
                section_config[section_config_var] = provider_val
        return section_config

    def set_default_provider(self, key, default_provider):
        provider = self._override_providers.get(key)
        if isinstance(provider, ChainProvider):
            provider.set_default_provider(default_provider)
            return
        elif isinstance(provider, BaseProvider):
            default_provider = ChainProvider(providers=[provider, default_provider])
        self._override_providers[key] = default_provider

    def __repr__(self):
        return f'SectionConfigProvider(section_name={self._section_name}, session={self._session}, override_providers={self._override_providers})'