from unittest import mock
import ddt
from cinderclient.tests.unit import utils
from cinderclient.v3 import limits
def _get_default_RateLimit(verb='verb1', uri='uri1', regex='regex1', value='value1', remain='remain1', unit='unit1', next_available='next1'):
    return limits.RateLimit(verb, uri, regex, value, remain, unit, next_available)