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
class KeyringBackend(metaclass=KeyringBackendMeta):
    """The abstract base class of the keyring, every backend must implement
    this interface.
    """

    def __init__(self):
        self.set_properties_from_env()

    @properties.classproperty
    def priority(self) -> float:
        """
        Each backend class must supply a priority, a number (float or integer)
        indicating the priority of the backend relative to all other backends.
        The priority need not be static -- it may (and should) vary based
        attributes of the environment in which is runs (platform, available
        packages, etc.).

        A higher number indicates a higher priority. The priority should raise
        a RuntimeError with a message indicating the underlying cause if the
        backend is not suitable for the current environment.

        As a rule of thumb, a priority between zero but less than one is
        suitable, but a priority of one or greater is recommended.
        """
        raise NotImplementedError

    @properties.classproperty
    def viable(cls):
        with errors.ExceptionRaisedContext() as exc:
            cls.priority
        return not exc

    @classmethod
    def get_viable_backends(cls: typing.Type[KeyringBackend]) -> filter[typing.Type[KeyringBackend]]:
        """
        Return all subclasses deemed viable.
        """
        return filter(operator.attrgetter('viable'), cls._classes)

    @properties.classproperty
    def name(cls) -> str:
        """
        The keyring name, suitable for display.

        The name is derived from module and class name.
        """
        parent, sep, mod_name = cls.__module__.rpartition('.')
        mod_name = mod_name.replace('_', ' ')
        return ' '.join([mod_name, cls.__name__])

    def __str__(self) -> str:
        keyring_class = type(self)
        return '{}.{} (priority: {:g})'.format(keyring_class.__module__, keyring_class.__name__, keyring_class.priority)

    @abc.abstractmethod
    def get_password(self, service: str, username: str) -> typing.Optional[str]:
        """Get password of the username for the service"""
        return None

    @abc.abstractmethod
    def set_password(self, service: str, username: str, password: str) -> None:
        """Set password for the username of the service.

        If the backend cannot store passwords, raise
        PasswordSetError.
        """
        raise errors.PasswordSetError('reason')

    def delete_password(self, service: str, username: str) -> None:
        """Delete the password for the username of the service.

        If the backend cannot delete passwords, raise
        PasswordDeleteError.
        """
        raise errors.PasswordDeleteError('reason')

    def get_credential(self, service: str, username: typing.Optional[str]) -> typing.Optional[credentials.Credential]:
        """Gets the username and password for the service.
        Returns a Credential instance.

        The *username* argument is optional and may be omitted by
        the caller or ignored by the backend. Callers must use the
        returned username.
        """
        if username is not None:
            password = self.get_password(service, username)
            if password is not None:
                return credentials.SimpleCredential(username, password)
        return None

    def set_properties_from_env(self) -> None:
        """For all KEYRING_PROPERTY_* env var, set that property."""

        def parse(item: typing.Tuple[str, str]):
            key, value = item
            pre, sep, name = key.partition('KEYRING_PROPERTY_')
            return sep and (name.lower(), value)
        props: filter[typing.Tuple[str, str]] = filter(None, map(parse, os.environ.items()))
        for name, value in props:
            setattr(self, name, value)

    def with_properties(self, **kwargs: typing.Any) -> KeyringBackend:
        alt = copy.copy(self)
        vars(alt).update(kwargs)
        return alt