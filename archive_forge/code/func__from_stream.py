import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
def _from_stream(self, stream):
    """Turns markup into a document.

        Just a wrapper around ElementTree which keeps track of namespaces.
        """
    events = ('start', 'start-ns', 'end-ns')
    root = None
    ns_map = []
    for event, elem in ET.iterparse(stream, events):
        if event == 'start-ns':
            ns_map.append(elem)
        elif event == 'end-ns':
            ns_map.pop()
        elif event == 'start':
            if root is None:
                root = elem
            elem.set(NS_MAP, dict(ns_map))
    return ET.ElementTree(root)