import re
from os import path
from ruamel import yaml
from kubernetes import client
class FailToCreateError(Exception):
    """
    An exception class for handling error if an error occurred when
    handling a yaml file.
    """

    def __init__(self, api_exceptions):
        self.api_exceptions = api_exceptions

    def __str__(self):
        msg = ''
        for api_exception in self.api_exceptions:
            msg += 'Error from server ({0}): {1}'.format(api_exception.reason, api_exception.body)
        return msg