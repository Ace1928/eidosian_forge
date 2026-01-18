import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
class Paginated(object):
    """Pretends to be a list if you iterate over it, but also keeps a
       next property you can use to get the next page of data.
    """

    def __init__(self, items=None, next_marker=None, links=None):
        self.items = items or []
        self.next = next_marker
        self.links = links or []

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return self.items.__iter__()

    def __getitem__(self, key):
        return self.items[key]

    def __setitem__(self, key, value):
        self.items[key] = value

    def __delitem__(self, key):
        del self.items[key]

    def __reversed__(self):
        return reversed(self.items)

    def __contains__(self, needle):
        return needle in self.items