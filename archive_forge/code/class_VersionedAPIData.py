from __future__ import absolute_import, division, print_function
from ansible_collections.community.routeros.plugins.module_utils.version import LooseVersion
class VersionedAPIData(object):

    def __init__(self, primary_keys=None, stratify_keys=None, required_one_of=None, mutually_exclusive=None, has_identifier=False, single_value=False, unknown_mechanism=False, fully_understood=False, fixed_entries=False, fields=None, versioned_fields=None):
        if sum([primary_keys is not None, stratify_keys is not None, has_identifier, single_value, unknown_mechanism]) > 1:
            raise ValueError('primary_keys, stratify_keys, has_identifier, single_value, and unknown_mechanism are mutually exclusive')
        if unknown_mechanism and fully_understood:
            raise ValueError('unknown_mechanism and fully_understood cannot be combined')
        self.primary_keys = primary_keys
        self.stratify_keys = stratify_keys
        self.required_one_of = required_one_of or []
        self.mutually_exclusive = mutually_exclusive or []
        self.has_identifier = has_identifier
        self.single_value = single_value
        self.unknown_mechanism = unknown_mechanism
        self.fully_understood = fully_understood
        self.fixed_entries = fixed_entries
        if fixed_entries and primary_keys is None:
            raise ValueError('fixed_entries can only be used with primary_keys')
        if fields is None:
            raise ValueError('fields must be provided')
        self.fields = fields
        if versioned_fields is not None:
            if not isinstance(versioned_fields, list):
                raise ValueError('unversioned_fields must be a list')
            for conditions, name, field in versioned_fields:
                if not isinstance(conditions, (tuple, list)):
                    raise ValueError('conditions must be a list or tuple')
                if not isinstance(field, KeyInfo):
                    raise ValueError('field must be a KeyInfo object')
                if name in fields:
                    raise ValueError('"{name}" appears both in fields and versioned_fields'.format(name=name))
        self.versioned_fields = versioned_fields or []
        if primary_keys:
            for pk in primary_keys:
                if pk not in fields:
                    raise ValueError('Primary key {pk} must be in fields!'.format(pk=pk))
        if stratify_keys:
            for sk in stratify_keys:
                if sk not in fields:
                    raise ValueError('Stratify key {sk} must be in fields!'.format(sk=sk))
        if required_one_of:
            for index, require_list in enumerate(required_one_of):
                if not isinstance(require_list, list):
                    raise ValueError('Require one of element at index #{index} must be a list!'.format(index=index + 1))
                for rk in require_list:
                    if rk not in fields:
                        raise ValueError('Require one of key {rk} must be in fields!'.format(rk=rk))
        if mutually_exclusive:
            for index, exclusive_list in enumerate(mutually_exclusive):
                if not isinstance(exclusive_list, list):
                    raise ValueError('Mutually exclusive element at index #{index} must be a list!'.format(index=index + 1))
                for ek in exclusive_list:
                    if ek not in fields:
                        raise ValueError('Mutually exclusive key {ek} must be in fields!'.format(ek=ek))
        self.needs_version = len(self.versioned_fields) > 0

    def specialize_for_version(self, api_version):
        fields = self.fields.copy()
        for conditions, name, field in self.versioned_fields:
            matching = True
            for other_version, comparator in conditions:
                other_api_version = LooseVersion(other_version)
                if not _compare(api_version, other_api_version, comparator):
                    matching = False
                    break
            if matching:
                if name in fields:
                    raise ValueError('Internal error: field "{field}" already exists for {version}'.format(field=name, version=api_version))
                fields[name] = field
        return VersionedAPIData(primary_keys=self.primary_keys, stratify_keys=self.stratify_keys, required_one_of=self.required_one_of, mutually_exclusive=self.mutually_exclusive, has_identifier=self.has_identifier, single_value=self.single_value, unknown_mechanism=self.unknown_mechanism, fully_understood=self.fully_understood, fixed_entries=self.fixed_entries, fields=fields)