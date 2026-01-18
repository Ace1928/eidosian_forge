import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
class SkipTemplate(Exception):
    """
    Raised to indicate that the template should not be copied over.
    Raise this exception during the substitution of your template
    """