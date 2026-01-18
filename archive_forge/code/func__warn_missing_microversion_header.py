import functools
import logging
import os
import pkgutil
import re
import traceback
import warnings
from oslo_utils import strutils
import novaclient
from novaclient import exceptions
from novaclient.i18n import _
def _warn_missing_microversion_header(header_name):
    """Log a warning about missing microversion response header."""
    LOG.warning(_('Your request was processed by a Nova API which does not support microversions (%s header is missing from response). Warning: Response may be incorrect.'), header_name)