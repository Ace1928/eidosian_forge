import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _strip_ns(self, name):
    """Strip the leading namespace from name.

        We don't have namespaces clashes in our context, stripping it makes the
        code simpler.
        """
    where = name.find(':')
    if where == -1:
        return name
    else:
        return name[where + 1:]