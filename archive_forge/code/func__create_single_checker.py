import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
def _create_single_checker(config):
    if 'response' in config['applies_when']:
        return _create_single_response_checker(config['applies_when']['response'])
    elif 'socket_errors' in config['applies_when']:
        return ExceptionRaiser()