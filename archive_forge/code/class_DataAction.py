import argparse
import warnings
from http.cookies import SimpleCookie
from shlex import split
from urllib.parse import urlparse
from w3lib.http import basic_auth_header
class DataAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        value = str(values)
        if value.startswith('$'):
            value = value[1:]
        setattr(namespace, self.dest, value)