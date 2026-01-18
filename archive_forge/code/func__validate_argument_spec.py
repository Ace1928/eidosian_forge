from __future__ import annotations
import abc
import argparse
import ast
import datetime
import json
import os
import re
import sys
import traceback
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from fnmatch import fnmatch
from antsibull_docs_parser import dom
from antsibull_docs_parser.parser import parse, Context
import yaml
from voluptuous.humanize import humanize_error
from ansible import __version__ as ansible_version
from ansible.executor.module_common import REPLACER_WINDOWS, NEW_STYLE_PYTHON_MODULE_RE
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.parameters import DEFAULT_TYPE_VALIDATORS
from ansible.module_utils.compat.version import StrictVersion, LooseVersion
from ansible.module_utils.basic import to_bytes
from ansible.module_utils.six import PY3, with_metaclass, string_types
from ansible.plugins.loader import fragment_loader
from ansible.plugins.list import IGNORE as REJECTLIST
from ansible.utils.plugin_docs import add_collection_to_versions_and_dates, add_fragments, get_docstring
from ansible.utils.version import SemanticVersion
from .module_args import AnsibleModuleImportError, AnsibleModuleNotInitialized, get_argument_spec
from .schema import (
from .utils import CaptureStd, NoArgsAnsibleModule, compare_unordered_lists, parse_yaml, parse_isodate
def _validate_argument_spec(self, docs, spec, kwargs, context=None, last_context_spec=None):
    if not self.analyze_arg_spec:
        return
    if docs is None:
        docs = {}
    if context is None:
        context = []
    if last_context_spec is None:
        last_context_spec = kwargs
    try:
        if not context:
            add_fragments(docs, self.object_path, fragment_loader=fragment_loader, is_module=self.plugin_type == 'module')
    except Exception:
        return
    module = NoArgsAnsibleModule({})
    self._validate_list_of_module_args('mutually_exclusive', last_context_spec.get('mutually_exclusive'), spec, context)
    self._validate_list_of_module_args('required_together', last_context_spec.get('required_together'), spec, context)
    self._validate_list_of_module_args('required_one_of', last_context_spec.get('required_one_of'), spec, context)
    self._validate_required_if(last_context_spec.get('required_if'), spec, context, module)
    self._validate_required_by(last_context_spec.get('required_by'), spec, context)
    provider_args = set()
    args_from_argspec = set()
    deprecated_args_from_argspec = set()
    doc_options = docs.get('options', {})
    if doc_options is None:
        doc_options = {}
    for arg, data in spec.items():
        restricted_argument_names = ('message', 'syslog_facility')
        if arg.lower() in restricted_argument_names:
            msg = "Argument '%s' in argument_spec " % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += 'must not be one of %s as it is used internally by Ansible Core Engine' % ','.join(restricted_argument_names)
            self.reporter.error(path=self.object_path, code='invalid-argument-name', msg=msg)
            continue
        if 'aliases' in data:
            for al in data['aliases']:
                if al.lower() in restricted_argument_names:
                    msg = "Argument alias '%s' in argument_spec " % al
                    if context:
                        msg += ' found in %s' % ' -> '.join(context)
                    msg += 'must not be one of %s as it is used internally by Ansible Core Engine' % ','.join(restricted_argument_names)
                    self.reporter.error(path=self.object_path, code='invalid-argument-name', msg=msg)
                    continue
        if all((data.get('no_log') is None, is_potential_secret_option(arg), data.get('type') not in ('path', 'bool'), data.get('choices') is None)):
            msg = "Argument '%s' in argument_spec could be a secret, though doesn't have `no_log` set" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            self.reporter.error(path=self.object_path, code='no-log-needed', msg=msg)
        if not isinstance(data, dict):
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' must be a dictionary/hash when used'
            self.reporter.error(path=self.object_path, code='invalid-argument-spec', msg=msg)
            continue
        removed_at_date = data.get('removed_at_date', None)
        if removed_at_date is not None:
            try:
                if parse_isodate(removed_at_date, allow_date=False) < datetime.date.today():
                    msg = "Argument '%s' in argument_spec" % arg
                    if context:
                        msg += ' found in %s' % ' -> '.join(context)
                    msg += " has a removed_at_date '%s' before today" % removed_at_date
                    self.reporter.error(path=self.object_path, code='deprecated-date', msg=msg)
            except ValueError:
                pass
        deprecated_aliases = data.get('deprecated_aliases', None)
        if deprecated_aliases is not None:
            for deprecated_alias in deprecated_aliases:
                if 'name' in deprecated_alias and 'date' in deprecated_alias:
                    try:
                        date = deprecated_alias['date']
                        if parse_isodate(date, allow_date=False) < datetime.date.today():
                            msg = "Argument '%s' in argument_spec" % arg
                            if context:
                                msg += ' found in %s' % ' -> '.join(context)
                            msg += " has deprecated aliases '%s' with removal date '%s' before today" % (deprecated_alias['name'], deprecated_alias['date'])
                            self.reporter.error(path=self.object_path, code='deprecated-date', msg=msg)
                    except ValueError:
                        pass
        has_version = False
        if self.collection and self.collection_version is not None:
            compare_version = self.collection_version
            version_of_what = 'this collection (%s)' % self.collection_version_str
            code_prefix = 'collection'
            has_version = True
        elif not self.collection:
            compare_version = LOOSE_ANSIBLE_VERSION
            version_of_what = 'Ansible (%s)' % ansible_version
            code_prefix = 'ansible'
            has_version = True
        removed_in_version = data.get('removed_in_version', None)
        if removed_in_version is not None:
            try:
                collection_name = data.get('removed_from_collection')
                removed_in = self._create_version(str(removed_in_version), collection_name=collection_name)
                if has_version and collection_name == self.collection_name and (compare_version >= removed_in):
                    msg = "Argument '%s' in argument_spec" % arg
                    if context:
                        msg += ' found in %s' % ' -> '.join(context)
                    msg += ' has a deprecated removed_in_version %r,' % removed_in_version
                    msg += ' i.e. the version is less than or equal to the current version of %s' % version_of_what
                    self.reporter.error(path=self.object_path, code=code_prefix + '-deprecated-version', msg=msg)
            except ValueError as e:
                msg = "Argument '%s' in argument_spec" % arg
                if context:
                    msg += ' found in %s' % ' -> '.join(context)
                msg += ' has an invalid removed_in_version number %r: %s' % (removed_in_version, e)
                self.reporter.error(path=self.object_path, code='invalid-deprecated-version', msg=msg)
            except TypeError:
                msg = "Argument '%s' in argument_spec" % arg
                if context:
                    msg += ' found in %s' % ' -> '.join(context)
                msg += ' has an invalid removed_in_version number %r: ' % (removed_in_version,)
                msg += ' error while comparing to version of %s' % version_of_what
                self.reporter.error(path=self.object_path, code='invalid-deprecated-version', msg=msg)
        if deprecated_aliases is not None:
            for deprecated_alias in deprecated_aliases:
                if 'name' in deprecated_alias and 'version' in deprecated_alias:
                    try:
                        collection_name = deprecated_alias.get('collection_name')
                        version = self._create_version(str(deprecated_alias['version']), collection_name=collection_name)
                        if has_version and collection_name == self.collection_name and (compare_version >= version):
                            msg = "Argument '%s' in argument_spec" % arg
                            if context:
                                msg += ' found in %s' % ' -> '.join(context)
                            msg += " has deprecated aliases '%s' with removal in version %r," % (deprecated_alias['name'], deprecated_alias['version'])
                            msg += ' i.e. the version is less than or equal to the current version of %s' % version_of_what
                            self.reporter.error(path=self.object_path, code=code_prefix + '-deprecated-version', msg=msg)
                    except ValueError as e:
                        msg = "Argument '%s' in argument_spec" % arg
                        if context:
                            msg += ' found in %s' % ' -> '.join(context)
                        msg += " has deprecated aliases '%s' with invalid removal version %r: %s" % (deprecated_alias['name'], deprecated_alias['version'], e)
                        self.reporter.error(path=self.object_path, code='invalid-deprecated-version', msg=msg)
                    except TypeError:
                        msg = "Argument '%s' in argument_spec" % arg
                        if context:
                            msg += ' found in %s' % ' -> '.join(context)
                        msg += " has deprecated aliases '%s' with invalid removal version %r:" % (deprecated_alias['name'], deprecated_alias['version'])
                        msg += ' error while comparing to version of %s' % version_of_what
                        self.reporter.error(path=self.object_path, code='invalid-deprecated-version', msg=msg)
        aliases = data.get('aliases', [])
        if arg in aliases:
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' is specified as its own alias'
            self.reporter.error(path=self.object_path, code='parameter-alias-self', msg=msg)
        if len(aliases) > len(set(aliases)):
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' has at least one alias specified multiple times in aliases'
            self.reporter.error(path=self.object_path, code='parameter-alias-repeated', msg=msg)
        if not context and arg == 'state':
            bad_states = set(['list', 'info', 'get']) & set(data.get('choices', set()))
            for bad_state in bad_states:
                self.reporter.error(path=self.object_path, code='parameter-state-invalid-choice', msg="Argument 'state' includes the value '%s' as a choice" % bad_state)
        if not data.get('removed_in_version', None) and (not data.get('removed_at_date', None)):
            args_from_argspec.add(arg)
            args_from_argspec.update(aliases)
        else:
            deprecated_args_from_argspec.add(arg)
            deprecated_args_from_argspec.update(aliases)
        if arg == 'provider' and self.object_path.startswith('lib/ansible/modules/network/'):
            if data.get('options') is not None and (not isinstance(data.get('options'), Mapping)):
                self.reporter.error(path=self.object_path, code='invalid-argument-spec-options', msg="Argument 'options' in argument_spec['provider'] must be a dictionary/hash when used")
            elif data.get('options'):
                for provider_arg, provider_data in data.get('options', {}).items():
                    provider_args.add(provider_arg)
                    provider_args.update(provider_data.get('aliases', []))
        if data.get('required') and data.get('default', object) != object:
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' is marked as required but specifies a default. Arguments with a default should not be marked as required'
            self.reporter.error(path=self.object_path, code='no-default-for-required-parameter', msg=msg)
        if arg in provider_args:
            continue
        _type = data.get('type', 'str')
        if callable(_type):
            _type_checker = _type
        else:
            _type_checker = DEFAULT_TYPE_VALIDATORS.get(_type)
        _elements = data.get('elements')
        if _type == 'list' and (not _elements):
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' defines type as list but elements is not defined'
            self.reporter.error(path=self.object_path, code='parameter-list-no-elements', msg=msg)
        if _elements:
            if not callable(_elements):
                DEFAULT_TYPE_VALIDATORS.get(_elements)
            if _type != 'list':
                msg = "Argument '%s' in argument_spec" % arg
                if context:
                    msg += ' found in %s' % ' -> '.join(context)
                msg += ' defines elements as %s but it is valid only when value of parameter type is list' % _elements
                self.reporter.error(path=self.object_path, code='parameter-invalid-elements', msg=msg)
        arg_default = None
        if 'default' in data and data['default'] is not None:
            try:
                with CaptureStd():
                    arg_default = _type_checker(data['default'])
            except (Exception, SystemExit):
                msg = "Argument '%s' in argument_spec" % arg
                if context:
                    msg += ' found in %s' % ' -> '.join(context)
                msg += ' defines default as (%r) but this is incompatible with parameter type %r' % (data['default'], _type)
                self.reporter.error(path=self.object_path, code='incompatible-default-type', msg=msg)
                continue
        doc_options_args = []
        for alias in sorted(set([arg] + list(aliases))):
            if alias in doc_options:
                doc_options_args.append(alias)
        if len(doc_options_args) == 0:
            doc_options_arg = {}
        else:
            doc_options_arg = doc_options[doc_options_args[0]]
            if len(doc_options_args) > 1:
                msg = "Argument '%s' in argument_spec" % arg
                if context:
                    msg += ' found in %s' % ' -> '.join(context)
                msg += ' with aliases %s is documented multiple times, namely as %s' % (', '.join(["'%s'" % alias for alias in aliases]), ', '.join(["'%s'" % alias for alias in doc_options_args]))
                self.reporter.error(path=self.object_path, code='parameter-documented-multiple-times', msg=msg)
        try:
            doc_default = None
            if 'default' in doc_options_arg and doc_options_arg['default'] is not None:
                with CaptureStd():
                    doc_default = _type_checker(doc_options_arg['default'])
        except (Exception, SystemExit):
            msg = "Argument '%s' in documentation" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' defines default as (%r) but this is incompatible with parameter type %r' % (doc_options_arg.get('default'), _type)
            self.reporter.error(path=self.object_path, code='doc-default-incompatible-type', msg=msg)
            continue
        if arg_default != doc_default:
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' defines default as (%r) but documentation defines default as (%r)' % (arg_default, doc_default)
            self.reporter.error(path=self.object_path, code='doc-default-does-not-match-spec', msg=msg)
        doc_type = doc_options_arg.get('type')
        if 'type' in data and data['type'] is not None:
            if doc_type is None:
                if not arg.startswith('_'):
                    msg = "Argument '%s' in argument_spec" % arg
                    if context:
                        msg += ' found in %s' % ' -> '.join(context)
                    msg += " defines type as %r but documentation doesn't define type" % data['type']
                    self.reporter.error(path=self.object_path, code='parameter-type-not-in-doc', msg=msg)
            elif data['type'] != doc_type:
                msg = "Argument '%s' in argument_spec" % arg
                if context:
                    msg += ' found in %s' % ' -> '.join(context)
                msg += ' defines type as %r but documentation defines type as %r' % (data['type'], doc_type)
                self.reporter.error(path=self.object_path, code='doc-type-does-not-match-spec', msg=msg)
        elif doc_type is None:
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += " uses default type ('str') but documentation doesn't define type"
            self.reporter.error(path=self.object_path, code='doc-missing-type', msg=msg)
        elif doc_type != 'str':
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += " implies type as 'str' but documentation defines as %r" % doc_type
            self.reporter.error(path=self.object_path, code='implied-parameter-type-mismatch', msg=msg)
        doc_choices = []
        try:
            for choice in doc_options_arg.get('choices', []):
                try:
                    with CaptureStd():
                        doc_choices.append(_type_checker(choice))
                except (Exception, SystemExit):
                    msg = "Argument '%s' in documentation" % arg
                    if context:
                        msg += ' found in %s' % ' -> '.join(context)
                    msg += ' defines choices as (%r) but this is incompatible with argument type %r' % (choice, _type)
                    self.reporter.error(path=self.object_path, code='doc-choices-incompatible-type', msg=msg)
                    raise StopIteration()
        except StopIteration:
            continue
        arg_choices = []
        try:
            for choice in data.get('choices', []):
                try:
                    with CaptureStd():
                        arg_choices.append(_type_checker(choice))
                except (Exception, SystemExit):
                    msg = "Argument '%s' in argument_spec" % arg
                    if context:
                        msg += ' found in %s' % ' -> '.join(context)
                    msg += ' defines choices as (%r) but this is incompatible with argument type %r' % (choice, _type)
                    self.reporter.error(path=self.object_path, code='incompatible-choices', msg=msg)
                    raise StopIteration()
        except StopIteration:
            continue
        if not compare_unordered_lists(arg_choices, doc_choices):
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' defines choices as (%r) but documentation defines choices as (%r)' % (arg_choices, doc_choices)
            self.reporter.error(path=self.object_path, code='doc-choices-do-not-match-spec', msg=msg)
        doc_required = doc_options_arg.get('required', False)
        data_required = data.get('required', False)
        if (doc_required or data_required) and (not (doc_required and data_required)):
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            if doc_required:
                msg += ' is not required, but is documented as being required'
            else:
                msg += ' is required, but is not documented as being required'
            self.reporter.error(path=self.object_path, code='doc-required-mismatch', msg=msg)
        doc_elements = doc_options_arg.get('elements', None)
        doc_type = doc_options_arg.get('type', 'str')
        data_elements = data.get('elements', None)
        if (doc_elements or data_elements) and (not doc_elements == data_elements):
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            if data_elements:
                msg += ' specifies elements as %s,' % data_elements
            else:
                msg += ' does not specify elements,'
            if doc_elements:
                msg += 'but elements is documented as being %s' % doc_elements
            else:
                msg += 'but elements is not documented'
            self.reporter.error(path=self.object_path, code='doc-elements-mismatch', msg=msg)
        spec_suboptions = data.get('options')
        doc_suboptions = doc_options_arg.get('suboptions', {})
        if spec_suboptions:
            if not doc_suboptions:
                msg = "Argument '%s' in argument_spec" % arg
                if context:
                    msg += ' found in %s' % ' -> '.join(context)
                msg += ' has sub-options but documentation does not define it'
                self.reporter.error(path=self.object_path, code='missing-suboption-docs', msg=msg)
            self._validate_argument_spec({'options': doc_suboptions}, spec_suboptions, kwargs, context=context + [arg], last_context_spec=data)
    for arg in args_from_argspec:
        if not str(arg).isidentifier():
            msg = "Argument '%s' in argument_spec" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' is not a valid python identifier'
            self.reporter.error(path=self.object_path, code='parameter-invalid', msg=msg)
    if docs:
        args_from_docs = set()
        for arg, data in doc_options.items():
            args_from_docs.add(arg)
            args_from_docs.update(data.get('aliases', []))
        args_missing_from_docs = args_from_argspec.difference(args_from_docs)
        docs_missing_from_args = args_from_docs.difference(args_from_argspec | deprecated_args_from_argspec)
        for arg in args_missing_from_docs:
            if arg in provider_args:
                continue
            msg = "Argument '%s'" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' is listed in the argument_spec, but not documented in the module documentation'
            self.reporter.error(path=self.object_path, code='undocumented-parameter', msg=msg)
        for arg in docs_missing_from_args:
            msg = "Argument '%s'" % arg
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' is listed in DOCUMENTATION.options, but not accepted by the module argument_spec'
            self.reporter.error(path=self.object_path, code='nonexistent-parameter-documented', msg=msg)