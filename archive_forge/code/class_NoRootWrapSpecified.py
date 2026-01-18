import functools
import logging
import multiprocessing
import os
import random
import shlex
import signal
import sys
import time
import warnings
import enum
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
class NoRootWrapSpecified(Exception):

    def __init__(self, message=None):
        super(NoRootWrapSpecified, self).__init__(message)