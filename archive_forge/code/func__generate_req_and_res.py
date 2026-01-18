import copy
import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallgroup
from neutronclient.osc.v2 import utils as v2_utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def _generate_req_and_res(verifylist):
    request = dict(verifylist)
    response = _fwg
    for key, val in verifylist:
        del request[key]
        if re.match('^no_', key) and val is True:
            new_value = None
        elif key == 'enable' and val:
            new_value = True
        elif key == 'disable' and val:
            new_value = False
        elif val is True or val is False:
            new_value = val
        elif key in ('name', 'description'):
            new_value = val
        else:
            new_value = val
        converted = CONVERT_MAP.get(key, key)
        request[converted] = new_value
        response[converted] = new_value
    return (request, response)