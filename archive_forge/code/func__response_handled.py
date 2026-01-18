import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _response_handled(self):
    """A response element inside a multistatus have been parsed."""
    if self.dir_content is None:
        self.dir_content = []
    self.dir_content.append(self._make_response_tuple())
    self._init_response_attrs()