import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
class ProfileConfigProvider(DatabricksConfigProvider):
    """Loads from the databrickscfg file."""

    def __init__(self, profile=DEFAULT_SECTION):
        self.profile = profile

    def get_config(self):
        raw_config = _fetch_from_fs()
        host = _get_option_if_exists(raw_config, self.profile, HOST)
        username = _get_option_if_exists(raw_config, self.profile, USERNAME)
        password = _get_option_if_exists(raw_config, self.profile, PASSWORD)
        token = _get_option_if_exists(raw_config, self.profile, TOKEN)
        refresh_token = _get_option_if_exists(raw_config, self.profile, REFRESH_TOKEN)
        insecure = _get_option_if_exists(raw_config, self.profile, INSECURE)
        jobs_api_version = _get_option_if_exists(raw_config, self.profile, JOBS_API_VERSION)
        config = DatabricksConfig(host, username, password, token, refresh_token, insecure, jobs_api_version)
        if config.is_valid:
            return config
        return None