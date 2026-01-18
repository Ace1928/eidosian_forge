from __future__ import with_statement
from logging import getLogger
import warnings
import sys
from passlib import hash, registry, exc
from passlib.registry import register_crypt_handler, register_crypt_handler_path, \
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase
class dummy_0(uh.StaticHandler):
    name = 'dummy_0'