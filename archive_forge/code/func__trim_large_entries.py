import logging
import os
import platform
import socket
import string
from base64 import b64encode
from urllib import parse
import certifi
import urllib3
from selenium import __version__
from . import utils
from .command import Command
from .errorhandler import ErrorCode
def _trim_large_entries(self, input_dict, max_length=100):
    """Truncate string values in a dictionary if they exceed max_length.

        :param dict: Dictionary with potentially large values
        :param max_length: Maximum allowed length of string values
        :return: Dictionary with truncated string values
        """
    output_dictionary = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            output_dictionary[key] = self._trim_large_entries(value, max_length)
        elif isinstance(value, str) and len(value) > max_length:
            output_dictionary[key] = value[:max_length] + '...'
        else:
            output_dictionary[key] = value
    return output_dictionary