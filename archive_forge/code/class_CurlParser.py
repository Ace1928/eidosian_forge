import argparse
import warnings
from http.cookies import SimpleCookie
from shlex import split
from urllib.parse import urlparse
from w3lib.http import basic_auth_header
class CurlParser(argparse.ArgumentParser):

    def error(self, message):
        error_msg = f'There was an error parsing the curl command: {message}'
        raise ValueError(error_msg)