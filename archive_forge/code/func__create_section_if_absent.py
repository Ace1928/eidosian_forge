import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
def _create_section_if_absent(raw_config, profile):
    if not raw_config.has_section(profile) and profile != DEFAULT_SECTION:
        raw_config.add_section(profile)