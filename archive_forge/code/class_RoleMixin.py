from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import pkgutil
import os
import os.path
import re
import textwrap
import traceback
import ansible.plugins.loader as plugin_loader
from pathlib import Path
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.collections.list import list_collection_dirs
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError, AnsiblePluginNotFound
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.json import json_dump
from ansible.module_utils.common.yaml import yaml_dump
from ansible.module_utils.compat import importlib
from ansible.module_utils.six import string_types
from ansible.parsing.plugin_docs import read_docstub
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.plugins.list import list_plugins
from ansible.plugins.loader import action_loader, fragment_loader
from ansible.utils.collection_loader import AnsibleCollectionConfig, AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_plugin_docs, get_docstring, get_versioned_doclink
class RoleMixin(object):
    """A mixin containing all methods relevant to role argument specification functionality.

    Note: The methods for actual display of role data are not present here.
    """
    ROLE_ARGSPEC_FILES = ['argument_specs' + e for e in C.YAML_FILENAME_EXTENSIONS] + ['main' + e for e in C.YAML_FILENAME_EXTENSIONS]

    def _load_argspec(self, role_name, collection_path=None, role_path=None):
        """Load the role argument spec data from the source file.

        :param str role_name: The name of the role for which we want the argspec data.
        :param str collection_path: Path to the collection containing the role. This
            will be None for standard roles.
        :param str role_path: Path to the standard role. This will be None for
            collection roles.

        We support two files containing the role arg spec data: either meta/main.yml
        or meta/argument_spec.yml. The argument_spec.yml file will take precedence
        over the meta/main.yml file, if it exists. Data is NOT combined between the
        two files.

        :returns: A dict of all data underneath the ``argument_specs`` top-level YAML
            key in the argspec data file. Empty dict is returned if there is no data.
        """
        if collection_path:
            meta_path = os.path.join(collection_path, 'roles', role_name, 'meta')
        elif role_path:
            meta_path = os.path.join(role_path, 'meta')
        else:
            raise AnsibleError("A path is required to load argument specs for role '%s'" % role_name)
        path = None
        for specfile in self.ROLE_ARGSPEC_FILES:
            full_path = os.path.join(meta_path, specfile)
            if os.path.exists(full_path):
                path = full_path
                break
        if path is None:
            return {}
        try:
            with open(path, 'r') as f:
                data = from_yaml(f.read(), file_name=path)
                if data is None:
                    data = {}
                return data.get('argument_specs', {})
        except (IOError, OSError) as e:
            raise AnsibleParserError("An error occurred while trying to read the file '%s': %s" % (path, to_native(e)), orig_exc=e)

    def _find_all_normal_roles(self, role_paths, name_filters=None):
        """Find all non-collection roles that have an argument spec file.

        Note that argument specs do not actually need to exist within the spec file.

        :param role_paths: A tuple of one or more role paths. When a role with the same name
            is found in multiple paths, only the first-found role is returned.
        :param name_filters: A tuple of one or more role names used to filter the results.

        :returns: A set of tuples consisting of: role name, full role path
        """
        found = set()
        found_names = set()
        for path in role_paths:
            if not os.path.isdir(path):
                continue
            for entry in os.listdir(path):
                role_path = os.path.join(path, entry)
                for specfile in self.ROLE_ARGSPEC_FILES:
                    full_path = os.path.join(role_path, 'meta', specfile)
                    if os.path.exists(full_path):
                        if name_filters is None or entry in name_filters:
                            if entry not in found_names:
                                found.add((entry, role_path))
                            found_names.add(entry)
                        break
        return found

    def _find_all_collection_roles(self, name_filters=None, collection_filter=None):
        """Find all collection roles with an argument spec file.

        Note that argument specs do not actually need to exist within the spec file.

        :param name_filters: A tuple of one or more role names used to filter the results. These
            might be fully qualified with the collection name (e.g., community.general.roleA)
            or not (e.g., roleA).

        :param collection_filter: A list of strings containing the FQCN of a collection which will
            be used to limit results. This filter will take precedence over the name_filters.

        :returns: A set of tuples consisting of: role name, collection name, collection path
        """
        found = set()
        b_colldirs = list_collection_dirs(coll_filter=collection_filter)
        for b_path in b_colldirs:
            path = to_text(b_path, errors='surrogate_or_strict')
            collname = _get_collection_name_from_path(b_path)
            roles_dir = os.path.join(path, 'roles')
            if os.path.exists(roles_dir):
                for entry in os.listdir(roles_dir):
                    for specfile in self.ROLE_ARGSPEC_FILES:
                        full_path = os.path.join(roles_dir, entry, 'meta', specfile)
                        if os.path.exists(full_path):
                            if name_filters is None:
                                found.add((entry, collname, path))
                            else:
                                for fqcn in name_filters:
                                    if len(fqcn.split('.')) == 3:
                                        ns, col, role = fqcn.split('.')
                                        if '.'.join([ns, col]) == collname and entry == role:
                                            found.add((entry, collname, path))
                                    elif fqcn == entry:
                                        found.add((entry, collname, path))
                            break
        return found

    def _build_summary(self, role, collection, argspec):
        """Build a summary dict for a role.

        Returns a simplified role arg spec containing only the role entry points and their
        short descriptions, and the role collection name (if applicable).

        :param role: The simple role name.
        :param collection: The collection containing the role (None or empty string if N/A).
        :param argspec: The complete role argspec data dict.

        :returns: A tuple with the FQCN role name and a summary dict.
        """
        if collection:
            fqcn = '.'.join([collection, role])
        else:
            fqcn = role
        summary = {}
        summary['collection'] = collection
        summary['entry_points'] = {}
        for ep in argspec.keys():
            entry_spec = argspec[ep] or {}
            summary['entry_points'][ep] = entry_spec.get('short_description', '')
        return (fqcn, summary)

    def _build_doc(self, role, path, collection, argspec, entry_point):
        if collection:
            fqcn = '.'.join([collection, role])
        else:
            fqcn = role
        doc = {}
        doc['path'] = path
        doc['collection'] = collection
        doc['entry_points'] = {}
        for ep in argspec.keys():
            if entry_point is None or ep == entry_point:
                entry_spec = argspec[ep] or {}
                doc['entry_points'][ep] = entry_spec
        if len(doc['entry_points'].keys()) == 0:
            doc = None
        return (fqcn, doc)

    def _create_role_list(self, fail_on_errors=True):
        """Return a dict describing the listing of all roles with arg specs.

        :param role_paths: A tuple of one or more role paths.

        :returns: A dict indexed by role name, with 'collection' and 'entry_points' keys per role.

        Example return:

            results = {
               'roleA': {
                  'collection': '',
                  'entry_points': {
                     'main': 'Short description for main'
                  }
               },
               'a.b.c.roleB': {
                  'collection': 'a.b.c',
                  'entry_points': {
                     'main': 'Short description for main',
                     'alternate': 'Short description for alternate entry point'
                  }
               'x.y.z.roleB': {
                  'collection': 'x.y.z',
                  'entry_points': {
                     'main': 'Short description for main',
                  }
               },
            }
        """
        roles_path = self._get_roles_path()
        collection_filter = self._get_collection_filter()
        if not collection_filter:
            roles = self._find_all_normal_roles(roles_path)
        else:
            roles = []
        collroles = self._find_all_collection_roles(collection_filter=collection_filter)
        result = {}
        for role, role_path in roles:
            try:
                argspec = self._load_argspec(role, role_path=role_path)
                fqcn, summary = self._build_summary(role, '', argspec)
                result[fqcn] = summary
            except Exception as e:
                if fail_on_errors:
                    raise
                result[role] = {'error': 'Error while loading role argument spec: %s' % to_native(e)}
        for role, collection, collection_path in collroles:
            try:
                argspec = self._load_argspec(role, collection_path=collection_path)
                fqcn, summary = self._build_summary(role, collection, argspec)
                result[fqcn] = summary
            except Exception as e:
                if fail_on_errors:
                    raise
                result['%s.%s' % (collection, role)] = {'error': 'Error while loading role argument spec: %s' % to_native(e)}
        return result

    def _create_role_doc(self, role_names, entry_point=None, fail_on_errors=True):
        """
        :param role_names: A tuple of one or more role names.
        :param role_paths: A tuple of one or more role paths.
        :param entry_point: A role entry point name for filtering.
        :param fail_on_errors: When set to False, include errors in the JSON output instead of raising errors

        :returns: A dict indexed by role name, with 'collection', 'entry_points', and 'path' keys per role.
        """
        roles_path = self._get_roles_path()
        roles = self._find_all_normal_roles(roles_path, name_filters=role_names)
        collroles = self._find_all_collection_roles(name_filters=role_names)
        result = {}
        for role, role_path in roles:
            try:
                argspec = self._load_argspec(role, role_path=role_path)
                fqcn, doc = self._build_doc(role, role_path, '', argspec, entry_point)
                if doc:
                    result[fqcn] = doc
            except Exception as e:
                result[role] = {'error': 'Error while processing role: %s' % to_native(e)}
        for role, collection, collection_path in collroles:
            try:
                argspec = self._load_argspec(role, collection_path=collection_path)
                fqcn, doc = self._build_doc(role, collection_path, collection, argspec, entry_point)
                if doc:
                    result[fqcn] = doc
            except Exception as e:
                result['%s.%s' % (collection, role)] = {'error': 'Error while processing role: %s' % to_native(e)}
        return result