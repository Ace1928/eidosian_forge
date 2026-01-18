import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _update_section_provider(self, config_store, section_name, variable, value):
    section_provider_copy = copy.deepcopy(config_store.get_config_provider(section_name))
    section_provider_copy.set_default_provider(variable, ConstantProvider(value))
    config_store.set_config_provider(section_name, section_provider_copy)