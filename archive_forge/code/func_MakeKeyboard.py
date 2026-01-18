from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def MakeKeyboard(path, usage):
    d = {}
    d['vendor_id'] = 1133
    d['product_id'] = 49948
    d['path'] = path
    d['usage'] = usage
    d['usage_page'] = 1
    return d