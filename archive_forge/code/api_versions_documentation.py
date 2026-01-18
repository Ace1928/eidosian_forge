import functools
import logging
import os
import pkgutil
import re
import traceback
from oslo_utils import strutils
from zunclient import exceptions
from zunclient.i18n import _
Versioning information for a single method

        :param name: Name of the method
        :param start_version: Minimum acceptable version
        :param end_version: Maximum acceptable_version
        :param func: Method to call

        Minimum and maximums are inclusive
        