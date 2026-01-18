from __future__ import annotations
import os
import abc
import logging
import operator
import copy
import typing
from .py312compat import metadata
from . import credentials, errors, util
from ._compat import properties
class SchemeSelectable:
    """
    Allow a backend to select different "schemes" for the
    username and service.

    >>> backend = SchemeSelectable()
    >>> backend._query('contoso', 'alice')
    {'username': 'alice', 'service': 'contoso'}
    >>> backend._query('contoso')
    {'service': 'contoso'}
    >>> backend.scheme = 'KeePassXC'
    >>> backend._query('contoso', 'alice')
    {'UserName': 'alice', 'Title': 'contoso'}
    >>> backend._query('contoso', 'alice', foo='bar')
    {'UserName': 'alice', 'Title': 'contoso', 'foo': 'bar'}
    """
    scheme = 'default'
    schemes = dict(default=dict(username='username', service='service'), KeePassXC=dict(username='UserName', service='Title'))

    def _query(self, service: str, username: typing.Optional[str]=None, **base: typing.Any) -> typing.Dict[str, str]:
        scheme = self.schemes[self.scheme]
        return dict({scheme['username']: username, scheme['service']: service} if username is not None else {scheme['service']: service}, **base)