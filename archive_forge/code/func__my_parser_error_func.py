import argparse
from oslotest import base
@staticmethod
def _my_parser_error_func(message):
    raise argparse.ArgumentTypeError(message)