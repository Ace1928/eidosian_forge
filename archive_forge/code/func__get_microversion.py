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
@classmethod
def _get_microversion(cls, session, *, action):
    """Get microversion to use for the given action.

        The base version uses the following logic:

        1. If the session has a default microversion for the current service,
           just use it.
        2. If ``self._max_microversion`` is not ``None``, use minimum between
           it and the maximum microversion supported by the server.
        3. Otherwise use ``None``.

        Subclasses can override this method if more complex logic is needed.

        :param session: The session to use for making the request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param action: One of "fetch", "commit", "create", "delete", "patch".
        :type action: str
        :return: Microversion as string or ``None``
        """
    if action not in {'list', 'fetch', 'commit', 'create', 'delete', 'patch'}:
        raise ValueError('Invalid action: %s' % action)
    if session.default_microversion:
        return session.default_microversion
    return utils.maximum_supported_microversion(session, cls._max_microversion)