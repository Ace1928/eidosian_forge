from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _ParseSelector(selector):
    """This function parses the selector flag."""
    if not selector:
        return (None, None)
    selectors = selector.split(',')
    selector_map = {}
    for s in selectors:
        items = s.split('=')
        if len(items) != 2:
            return (None, '--selector should have the format key1=value1,key2=value2')
        selector_map[items[0]] = items[1]
    return (selector_map, None)