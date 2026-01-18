import abc
import collections
import inspect
import itertools
import operator
import typing as ty
import urllib.parse
import warnings
import jsonpatch
from keystoneauth1 import adapter
from keystoneauth1 import discover
from requests import structures
from openstack import _log
from openstack import exceptions
from openstack import format
from openstack import utils
from openstack import warnings as os_warnings
def _attr_to_dict(self, attr, to_munch):
    """For a given attribute, convert it into a form suitable for a dict
        value.

        :param bool attr: Attribute name to convert

        :return: A dictionary of key/value pairs where keys are named
            as they exist as attributes of this class.
        :param bool _to_munch: Converts subresources to munch instead of dict.
        """
    value = getattr(self, attr, None)
    if isinstance(value, Resource):
        return value.to_dict(_to_munch=to_munch)
    elif isinstance(value, dict) and to_munch:
        return utils.Munch(value)
    elif value and isinstance(value, list):
        converted = []
        for raw in value:
            if isinstance(raw, Resource):
                converted.append(raw.to_dict(_to_munch=to_munch))
            elif isinstance(raw, dict) and to_munch:
                converted.append(utils.Munch(raw))
            else:
                converted.append(raw)
        return converted
    return value