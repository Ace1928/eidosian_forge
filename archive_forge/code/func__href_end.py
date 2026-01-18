import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _href_end(self):
    stack = self.elt_stack
    return len(stack) == 3 and stack[0] == 'multistatus' and (stack[1] == 'response') and (stack[2] == 'href')