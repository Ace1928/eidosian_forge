from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import add_metaclass
class _AnsibleCollectionConfig(type):

    def __init__(cls, meta, name, bases):
        cls._collection_finder = None
        cls._default_collection = None
        cls._on_collection_load = _EventSource()

    @property
    def collection_finder(cls):
        return cls._collection_finder

    @collection_finder.setter
    def collection_finder(cls, value):
        if cls._collection_finder:
            raise ValueError('an AnsibleCollectionFinder has already been configured')
        cls._collection_finder = value

    @property
    def collection_paths(cls):
        cls._require_finder()
        return [to_text(p) for p in cls._collection_finder._n_collection_paths]

    @property
    def default_collection(cls):
        return cls._default_collection

    @default_collection.setter
    def default_collection(cls, value):
        cls._default_collection = value

    @property
    def on_collection_load(cls):
        return cls._on_collection_load

    @on_collection_load.setter
    def on_collection_load(cls, value):
        if value is not cls._on_collection_load:
            raise ValueError('on_collection_load is not directly settable (use +=)')

    @property
    def playbook_paths(cls):
        cls._require_finder()
        return [to_text(p) for p in cls._collection_finder._n_playbook_paths]

    @playbook_paths.setter
    def playbook_paths(cls, value):
        cls._require_finder()
        cls._collection_finder.set_playbook_paths(value)

    def _require_finder(cls):
        if not cls._collection_finder:
            raise NotImplementedError('an AnsibleCollectionFinder has not been installed in this process')