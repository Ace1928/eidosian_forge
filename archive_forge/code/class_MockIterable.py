import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
class MockIterable(collections.abc.Iterable):
    links = ['item']
    count = 1

    def __iter__(self):
        return ['item1']

    def __len__(self):
        return self.count