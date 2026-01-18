import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
def _fetch_from_fs():
    raw_config = ConfigParser()
    raw_config.read(_get_path())
    return raw_config