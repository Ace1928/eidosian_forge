from __future__ import annotations
import os
import sys
from typing import Any
import click
import streamlit.runtime.caching as caching
import streamlit.runtime.legacy_caching as legacy_caching
import streamlit.web.bootstrap as bootstrap
from streamlit import config as _config
from streamlit.config_option import ConfigOption
from streamlit.runtime.credentials import Credentials, check_credentials
from streamlit.web.cache_storage_manager_config import (
def _convert_config_option_to_click_option(config_option: ConfigOption) -> dict[str, Any]:
    """Composes given config option options as options for click lib."""
    option = f'--{config_option.key}'
    param = config_option.key.replace('.', '_')
    description = config_option.description
    if config_option.deprecated:
        if description is None:
            description = ''
        description += f'\n {config_option.deprecation_text} - {config_option.expiration_date}'
    return {'param': param, 'description': description, 'type': config_option.type, 'option': option, 'envvar': config_option.env_var}