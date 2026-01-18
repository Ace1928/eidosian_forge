from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def RPad(collection, to_size):
    while len(collection) < to_size:
        collection.append(0)
    return collection