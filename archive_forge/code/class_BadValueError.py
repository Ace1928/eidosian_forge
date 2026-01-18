import os
import re
import warnings
from googlecloudsdk.third_party.appengine.api import lib_config
class BadValueError(Exception):
    """Raised by ValidateNamespaceString."""