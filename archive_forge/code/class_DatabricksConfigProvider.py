import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
class DatabricksConfigProvider:
    """
    Responsible for providing hostname and authentication information to make
    API requests against the Databricks REST API.
    This method should generally return None if it cannot provide credentials, in order
    to facilitate chanining of providers.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_config(self):
        pass