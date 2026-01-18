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
class ConfWithId(BaseConf):
    """Configuration settings related to identity."""
    ID = 'id'
    NAME = 'name'
    DESCRIPTION = 'description'
    UPDATE_NAME_EVT = 'update_name_evt'
    UPDATE_DESCRIPTION_EVT = 'update_description_evt'
    VALID_EVT = frozenset([UPDATE_NAME_EVT, UPDATE_DESCRIPTION_EVT])
    OPTIONAL_SETTINGS = frozenset([ID, NAME, DESCRIPTION])

    def __init__(self, **kwargs):
        super(ConfWithId, self).__init__(**kwargs)

    @classmethod
    def get_opt_settings(cls):
        self_confs = super(ConfWithId, cls).get_opt_settings()
        self_confs.update(ConfWithId.OPTIONAL_SETTINGS)
        return self_confs

    @classmethod
    def get_req_settings(cls):
        self_confs = super(ConfWithId, cls).get_req_settings()
        return self_confs

    @classmethod
    def get_valid_evts(cls):
        self_valid_evts = super(ConfWithId, cls).get_valid_evts()
        self_valid_evts.update(ConfWithId.VALID_EVT)
        return self_valid_evts

    def _init_opt_settings(self, **kwargs):
        super(ConfWithId, self)._init_opt_settings(**kwargs)
        self._settings[ConfWithId.ID] = compute_optional_conf(ConfWithId.ID, str(uuid.uuid4()), **kwargs)
        self._settings[ConfWithId.NAME] = compute_optional_conf(ConfWithId.NAME, str(self), **kwargs)
        self._settings[ConfWithId.DESCRIPTION] = compute_optional_conf(ConfWithId.DESCRIPTION, str(self), **kwargs)

    @property
    def id(self):
        return self._settings[ConfWithId.ID]

    @property
    def name(self):
        return self._settings[ConfWithId.NAME]

    @name.setter
    def name(self, new_name):
        old_name = self.name
        if not new_name:
            new_name = repr(self)
        else:
            get_validator(ConfWithId.NAME)(new_name)
        if old_name != new_name:
            self._settings[ConfWithId.NAME] = new_name
            self._notify_listeners(ConfWithId.UPDATE_NAME_EVT, (old_name, self.name))

    @property
    def description(self):
        return self._settings[ConfWithId.DESCRIPTION]

    @description.setter
    def description(self, new_description):
        old_desc = self.description
        if not new_description:
            new_description = str(self)
        else:
            get_validator(ConfWithId.DESCRIPTION)(new_description)
        if old_desc != new_description:
            self._settings[ConfWithId.DESCRIPTION] = new_description
            self._notify_listeners(ConfWithId.UPDATE_DESCRIPTION_EVT, (old_desc, self.description))

    def update(self, **kwargs):
        super(ConfWithId, self).update(**kwargs)
        self.name = compute_optional_conf(ConfWithId.NAME, str(self), **kwargs)
        self.description = compute_optional_conf(ConfWithId.DESCRIPTION, str(self), **kwargs)