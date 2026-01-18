import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
def _get_option_if_exists(raw_config, profile, option):
    if profile == DEFAULT_SECTION:
        return raw_config.get(profile, option) if raw_config.has_option(profile, option) else None
    elif option not in raw_config._sections.get(profile, {}).keys():
        return None
    return raw_config.get(profile, option)