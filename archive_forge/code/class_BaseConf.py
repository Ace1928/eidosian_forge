import abc
from abc import ABCMeta
from abc import abstractmethod
import functools
import numbers
import logging
import uuid
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.base import RUNTIME_CONF_ERROR_CODE
from os_ken.services.protocols.bgp.base import validate
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
class BaseConf(object, metaclass=abc.ABCMeta):
    """Base class for a set of configuration values.

    Configurations can be required or optional. Also acts as a container of
    configuration change listeners.
    """

    def __init__(self, **kwargs):
        self._req_settings = self.get_req_settings()
        self._opt_settings = self.get_opt_settings()
        self._valid_evts = self.get_valid_evts()
        self._listeners = {}
        self._settings = {}
        self._validate_req_unknown_settings(**kwargs)
        self._init_req_settings(**kwargs)
        self._init_opt_settings(**kwargs)

    @property
    def settings(self):
        """Returns a copy of current settings."""
        return self._settings.copy()

    @classmethod
    def get_valid_evts(cls):
        return set()

    @classmethod
    def get_req_settings(cls):
        return set()

    @classmethod
    def get_opt_settings(cls):
        return set()

    @abstractmethod
    def _init_opt_settings(self, **kwargs):
        """Sub-classes should override this method to initialize optional
         settings.
        """
        pass

    @abstractmethod
    def update(self, **kwargs):
        self._validate_req_unknown_settings(**kwargs)

    def _validate_req_unknown_settings(self, **kwargs):
        """Checks if required settings are present.

        Also checks if unknown requirements are present.
        """
        self._all_attrs = self._req_settings | self._opt_settings
        if not kwargs and len(self._req_settings) > 0:
            raise MissingRequiredConf(desc='Missing all required attributes.')
        given_attrs = frozenset(kwargs.keys())
        unknown_attrs = given_attrs - self._all_attrs
        if unknown_attrs:
            raise RuntimeConfigError(desc='Unknown attributes: %s' % ', '.join([str(i) for i in unknown_attrs]))
        missing_req_settings = self._req_settings - given_attrs
        if missing_req_settings:
            raise MissingRequiredConf(conf_name=list(missing_req_settings))

    def _init_req_settings(self, **kwargs):
        for req_attr in self._req_settings:
            req_attr_value = kwargs.get(req_attr)
            if req_attr_value is None:
                raise MissingRequiredConf(conf_name=req_attr_value)
            req_attr_value = get_validator(req_attr)(req_attr_value)
            self._settings[req_attr] = req_attr_value

    def add_listener(self, evt, callback):
        listeners = self._listeners.get(evt, None)
        if not listeners:
            listeners = set()
            self._listeners[evt] = listeners
        listeners.update([callback])

    def remove_listener(self, evt, callback):
        if evt in self.get_valid_evts():
            listeners = self._listeners.get(evt, None)
            if listeners and callback in listeners:
                listeners.remove(callback)
                return True
        return False

    def _notify_listeners(self, evt, value):
        listeners = self._listeners.get(evt, [])
        for callback in listeners:
            callback(ConfEvent(self, evt, value))

    def __repr__(self):
        return '%s(%r)' % (self.__class__, self._settings)