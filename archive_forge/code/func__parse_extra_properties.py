import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
def _parse_extra_properties(self, extra_properties):
    result = {}
    if extra_properties:
        for _property in extra_properties:
            result[_property['name']] = None
    return result