import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
def _require(self, *params):
    for param in params:
        if not hasattr(self, param):
            raise ArgumentRequired(param)
        if not getattr(self, param):
            raise ArgumentRequired(param)