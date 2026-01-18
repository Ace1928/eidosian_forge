import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _curry(*args, **kwargs):
    function, args = (args[0], args[1:])

    def result(*rest, **kwrest):
        combined = kwargs.copy()
        combined.update(kwrest)
        return function(*args + rest, **combined)
    return result