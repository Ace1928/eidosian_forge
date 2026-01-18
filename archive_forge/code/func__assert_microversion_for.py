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
def _assert_microversion_for(self, session, action, expected, error_message=None, maximum=None):
    """Enforce that the microversion for action satisfies the requirement.

        :param session: :class`keystoneauth1.adapter.Adapter`
        :param action: One of "fetch", "commit", "create", "delete".
        :param expected: Expected microversion.
        :param error_message: Optional error message with details. Will be
            prepended to the message generated here.
        :param maximum: Maximum microversion.
        :return: resulting microversion as string.
        :raises: :exc:`~openstack.exceptions.NotSupported` if the version
            used for the action is lower than the expected one.
        """

    def _raise(message):
        if error_message:
            error_message.rstrip('.')
            message = '%s. %s' % (error_message, message)
        raise exceptions.NotSupported(message)
    actual = self._get_microversion(session, action=action)
    if actual is None:
        message = 'API version %s is required, but the default version will be used.' % expected
        _raise(message)
    actual_n = discover.normalize_version_number(actual)
    if expected is not None:
        expected_n = discover.normalize_version_number(expected)
        if actual_n < expected_n:
            message = 'API version %(expected)s is required, but %(actual)s will be used.' % {'expected': expected, 'actual': actual}
            _raise(message)
    if maximum is not None:
        maximum_n = discover.normalize_version_number(maximum)
        if actual_n > maximum_n:
            return maximum
    return actual