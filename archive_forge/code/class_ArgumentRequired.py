import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
class ArgumentRequired(Exception):

    def __init__(self, param):
        self.param = param

    def __str__(self):
        return 'Argument "--%s" required.' % self.param