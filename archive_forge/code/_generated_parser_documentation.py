import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
Parse the given text, according to the options provided.

        The 'start' parameter is required if Lark was given multiple possible start symbols (using the start option).

        Returns a tree, unless specified otherwise.
        