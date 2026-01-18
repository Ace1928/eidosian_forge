import logging
import re
import sys
from ast import literal_eval as numeric
from .std import TqdmKeyError, TqdmTypeError, tqdm
from .version import __version__
class stdout(object):

    @staticmethod
    def write(x):
        with tqdm.external_write_mode(file=fp):
            fp_write(x)
        stdout_write(x)