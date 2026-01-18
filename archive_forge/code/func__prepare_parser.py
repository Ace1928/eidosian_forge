import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
@classmethod
def _prepare_parser(cls, parser):
    for param in cls.params:
        parser.add_option('--%s' % param)