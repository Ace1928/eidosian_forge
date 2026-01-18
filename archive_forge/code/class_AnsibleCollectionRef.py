from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
class AnsibleCollectionRef:
    VALID_REF_TYPES = frozenset((to_text(r) for r in ['action', 'become', 'cache', 'callback', 'cliconf', 'connection', 'doc_fragments', 'filter', 'httpapi', 'inventory', 'lookup', 'module_utils', 'modules', 'netconf', 'role', 'shell', 'strategy', 'terminal', 'test', 'vars', 'playbook']))
    VALID_SUBDIRS_RE = re.compile(to_text('^\\w+(\\.\\w+)*$'))
    VALID_FQCR_RE = re.compile(to_text('^\\w+(\\.\\w+){2,}$'))

    def __init__(self, collection_name, subdirs, resource, ref_type):
        """
        Create an AnsibleCollectionRef from components
        :param collection_name: a collection name of the form 'namespace.collectionname'
        :param subdirs: optional subdir segments to be appended below the plugin type (eg, 'subdir1.subdir2')
        :param resource: the name of the resource being references (eg, 'mymodule', 'someaction', 'a_role')
        :param ref_type: the type of the reference, eg 'module', 'role', 'doc_fragment'
        """
        collection_name = to_text(collection_name, errors='strict')
        if subdirs is not None:
            subdirs = to_text(subdirs, errors='strict')
        resource = to_text(resource, errors='strict')
        ref_type = to_text(ref_type, errors='strict')
        if not self.is_valid_collection_name(collection_name):
            raise ValueError('invalid collection name (must be of the form namespace.collection): {0}'.format(to_native(collection_name)))
        if ref_type not in self.VALID_REF_TYPES:
            raise ValueError('invalid collection ref_type: {0}'.format(ref_type))
        self.collection = collection_name
        if subdirs:
            if not re.match(self.VALID_SUBDIRS_RE, subdirs):
                raise ValueError('invalid subdirs entry: {0} (must be empty/None or of the form subdir1.subdir2)'.format(to_native(subdirs)))
            self.subdirs = subdirs
        else:
            self.subdirs = u''
        self.resource = resource
        self.ref_type = ref_type
        package_components = [u'ansible_collections', self.collection]
        fqcr_components = [self.collection]
        self.n_python_collection_package_name = to_native('.'.join(package_components))
        if self.ref_type == u'role':
            package_components.append(u'roles')
        elif self.ref_type == u'playbook':
            package_components.append(u'playbooks')
        else:
            package_components += [u'plugins', self.ref_type]
        if self.subdirs:
            package_components.append(self.subdirs)
            fqcr_components.append(self.subdirs)
        if self.ref_type in (u'role', u'playbook'):
            package_components.append(self.resource)
        fqcr_components.append(self.resource)
        self.n_python_package_name = to_native('.'.join(package_components))
        self._fqcr = u'.'.join(fqcr_components)

    def __repr__(self):
        return 'AnsibleCollectionRef(collection={0!r}, subdirs={1!r}, resource={2!r})'.format(self.collection, self.subdirs, self.resource)

    @property
    def fqcr(self):
        return self._fqcr

    @staticmethod
    def from_fqcr(ref, ref_type):
        """
        Parse a string as a fully-qualified collection reference, raises ValueError if invalid
        :param ref: collection reference to parse (a valid ref is of the form 'ns.coll.resource' or 'ns.coll.subdir1.subdir2.resource')
        :param ref_type: the type of the reference, eg 'module', 'role', 'doc_fragment'
        :return: a populated AnsibleCollectionRef object
        """
        if not AnsibleCollectionRef.is_valid_fqcr(ref):
            raise ValueError('{0} is not a valid collection reference'.format(to_native(ref)))
        ref = to_text(ref, errors='strict')
        ref_type = to_text(ref_type, errors='strict')
        ext = ''
        if ref_type == u'playbook' and ref.endswith(PB_EXTENSIONS):
            resource_splitname = ref.rsplit(u'.', 2)
            package_remnant = resource_splitname[0]
            resource = resource_splitname[1]
            ext = '.' + resource_splitname[2]
        else:
            resource_splitname = ref.rsplit(u'.', 1)
            package_remnant = resource_splitname[0]
            resource = resource_splitname[1]
        package_splitname = package_remnant.split(u'.', 2)
        if len(package_splitname) == 3:
            subdirs = package_splitname[2]
        else:
            subdirs = u''
        collection_name = u'.'.join(package_splitname[0:2])
        return AnsibleCollectionRef(collection_name, subdirs, resource + ext, ref_type)

    @staticmethod
    def try_parse_fqcr(ref, ref_type):
        """
        Attempt to parse a string as a fully-qualified collection reference, returning None on failure (instead of raising an error)
        :param ref: collection reference to parse (a valid ref is of the form 'ns.coll.resource' or 'ns.coll.subdir1.subdir2.resource')
        :param ref_type: the type of the reference, eg 'module', 'role', 'doc_fragment'
        :return: a populated AnsibleCollectionRef object on successful parsing, else None
        """
        try:
            return AnsibleCollectionRef.from_fqcr(ref, ref_type)
        except ValueError:
            pass

    @staticmethod
    def legacy_plugin_dir_to_plugin_type(legacy_plugin_dir_name):
        """
        Utility method to convert from a PluginLoader dir name to a plugin ref_type
        :param legacy_plugin_dir_name: PluginLoader dir name (eg, 'action_plugins', 'library')
        :return: the corresponding plugin ref_type (eg, 'action', 'role')
        """
        legacy_plugin_dir_name = to_text(legacy_plugin_dir_name)
        plugin_type = legacy_plugin_dir_name.removesuffix(u'_plugins')
        if plugin_type == u'library':
            plugin_type = u'modules'
        if plugin_type not in AnsibleCollectionRef.VALID_REF_TYPES:
            raise ValueError('{0} cannot be mapped to a valid collection ref type'.format(to_native(legacy_plugin_dir_name)))
        return plugin_type

    @staticmethod
    def is_valid_fqcr(ref, ref_type=None):
        """
        Validates if is string is a well-formed fully-qualified collection reference (does not look up the collection itself)
        :param ref: candidate collection reference to validate (a valid ref is of the form 'ns.coll.resource' or 'ns.coll.subdir1.subdir2.resource')
        :param ref_type: optional reference type to enable deeper validation, eg 'module', 'role', 'doc_fragment'
        :return: True if the collection ref passed is well-formed, False otherwise
        """
        ref = to_text(ref)
        if not ref_type:
            return bool(re.match(AnsibleCollectionRef.VALID_FQCR_RE, ref))
        return bool(AnsibleCollectionRef.try_parse_fqcr(ref, ref_type))

    @staticmethod
    def is_valid_collection_name(collection_name):
        """
        Validates if the given string is a well-formed collection name (does not look up the collection itself)
        :param collection_name: candidate collection name to validate (a valid name is of the form 'ns.collname')
        :return: True if the collection name passed is well-formed, False otherwise
        """
        collection_name = to_text(collection_name)
        if collection_name.count(u'.') != 1:
            return False
        return all((not iskeyword(ns_or_name) and is_python_identifier(ns_or_name) for ns_or_name in collection_name.split(u'.')))