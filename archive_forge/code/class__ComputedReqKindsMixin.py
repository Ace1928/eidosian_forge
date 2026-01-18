from __future__ import (absolute_import, division, print_function)
import os
import typing as t
from collections import namedtuple
from collections.abc import MutableSequence, MutableMapping
from glob import iglob
from urllib.parse import urlparse
from yaml import safe_load
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection import HAS_PACKAGING, PkgReq
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
class _ComputedReqKindsMixin:
    UNIQUE_ATTRS = ('fqcn', 'ver', 'src', 'type')

    def __init__(self, *args, **kwargs):
        if not self.may_have_offline_galaxy_info:
            self._source_info = None
        else:
            info_path = self.construct_galaxy_info_path(to_bytes(self.src, errors='surrogate_or_strict'))
            self._source_info = get_validated_source_info(info_path, self.namespace, self.name, self.ver)

    def __hash__(self):
        return hash(tuple((getattr(self, attr) for attr in _ComputedReqKindsMixin.UNIQUE_ATTRS)))

    def __eq__(self, candidate):
        return hash(self) == hash(candidate)

    @classmethod
    def from_dir_path_as_unknown(cls, dir_path, art_mgr):
        """Make collection from an unspecified dir type.

        This alternative constructor attempts to grab metadata from the
        given path if it's a directory. If there's no metadata, it
        falls back to guessing the FQCN based on the directory path and
        sets the version to "*".

        It raises a ValueError immediately if the input is not an
        existing directory path.
        """
        if not os.path.isdir(dir_path):
            raise ValueError("The collection directory '{path!s}' doesn't exist".format(path=to_native(dir_path)))
        try:
            return cls.from_dir_path(dir_path, art_mgr)
        except ValueError:
            return cls.from_dir_path_implicit(dir_path)

    @classmethod
    def from_dir_path(cls, dir_path, art_mgr):
        """Make collection from an directory with metadata."""
        if dir_path.endswith(to_bytes(os.path.sep)):
            dir_path = dir_path.rstrip(to_bytes(os.path.sep))
        if not _is_collection_dir(dir_path):
            display.warning(u"Collection at '{path!s}' does not have a {manifest_json!s} file, nor has it {galaxy_yml!s}: cannot detect version.".format(galaxy_yml=to_text(_GALAXY_YAML), manifest_json=to_text(_MANIFEST_JSON), path=to_text(dir_path, errors='surrogate_or_strict')))
            raise ValueError('`dir_path` argument must be an installed or a source collection directory.')
        tmp_inst_req = cls(None, None, dir_path, 'dir', None)
        req_version = art_mgr.get_direct_collection_version(tmp_inst_req)
        try:
            req_name = art_mgr.get_direct_collection_fqcn(tmp_inst_req)
        except TypeError as err:
            display.warning(u"Collection at '{path!s}' has a {manifest_json!s} or {galaxy_yml!s} file but it contains invalid metadata.".format(galaxy_yml=to_text(_GALAXY_YAML), manifest_json=to_text(_MANIFEST_JSON), path=to_text(dir_path, errors='surrogate_or_strict')))
            raise ValueError("Collection at '{path!s}' has invalid metadata".format(path=to_text(dir_path, errors='surrogate_or_strict'))) from err
        return cls(req_name, req_version, dir_path, 'dir', None)

    @classmethod
    def from_dir_path_implicit(cls, dir_path):
        """Construct a collection instance based on an arbitrary dir.

        This alternative constructor infers the FQCN based on the parent
        and current directory names. It also sets the version to "*"
        regardless of whether any of known metadata files are present.
        """
        if dir_path.endswith(to_bytes(os.path.sep)):
            dir_path = dir_path.rstrip(to_bytes(os.path.sep))
        u_dir_path = to_text(dir_path, errors='surrogate_or_strict')
        path_list = u_dir_path.split(os.path.sep)
        req_name = '.'.join(path_list[-2:])
        return cls(req_name, '*', dir_path, 'dir', None)

    @classmethod
    def from_string(cls, collection_input, artifacts_manager, supplemental_signatures):
        req = {}
        if _is_concrete_artifact_pointer(collection_input) or AnsibleCollectionRef.is_valid_collection_name(collection_input):
            req['name'] = collection_input
        elif ':' in collection_input:
            req['name'], _sep, req['version'] = collection_input.partition(':')
            if not req['version']:
                del req['version']
        else:
            if not HAS_PACKAGING:
                raise AnsibleError('Failed to import packaging, check that a supported version is installed')
            try:
                pkg_req = PkgReq(collection_input)
            except Exception as e:
                req['name'] = collection_input
            else:
                req['name'] = pkg_req.name
                if pkg_req.specifier:
                    req['version'] = to_text(pkg_req.specifier)
        req['signatures'] = supplemental_signatures
        return cls.from_requirement_dict(req, artifacts_manager)

    @classmethod
    def from_requirement_dict(cls, collection_req, art_mgr, validate_signature_options=True):
        req_name = collection_req.get('name', None)
        req_version = collection_req.get('version', '*')
        req_type = collection_req.get('type')
        req_source = collection_req.get('source', None)
        req_signature_sources = collection_req.get('signatures', None)
        if req_signature_sources is not None:
            if validate_signature_options and art_mgr.keyring is None:
                raise AnsibleError(f'Signatures were provided to verify {req_name} but no keyring was configured.')
            if not isinstance(req_signature_sources, MutableSequence):
                req_signature_sources = [req_signature_sources]
            req_signature_sources = frozenset(req_signature_sources)
        if req_type is None:
            if _ALLOW_CONCRETE_POINTER_IN_SOURCE and req_source is not None and _is_concrete_artifact_pointer(req_source):
                src_path = req_source
            elif req_name is not None and AnsibleCollectionRef.is_valid_collection_name(req_name):
                req_type = 'galaxy'
            elif req_name is not None and _is_concrete_artifact_pointer(req_name):
                src_path, req_name = (req_name, None)
            else:
                dir_tip_tmpl = '\n\nTip: Make sure you are pointing to the right subdirectory — `{src!s}` looks like a directory but it is neither a collection, nor a namespace dir.'
                if req_source is not None and os.path.isdir(req_source):
                    tip = dir_tip_tmpl.format(src=req_source)
                elif req_name is not None and os.path.isdir(req_name):
                    tip = dir_tip_tmpl.format(src=req_name)
                elif req_name:
                    tip = '\n\nCould not find {0}.'.format(req_name)
                else:
                    tip = ''
                raise AnsibleError("Neither the collection requirement entry key 'name', nor 'source' point to a concrete resolvable collection artifact. Also 'name' is not an FQCN. A valid collection name must be in the format <namespace>.<collection>. Please make sure that the namespace and the collection name contain characters from [a-zA-Z0-9_] only.{extra_tip!s}".format(extra_tip=tip))
        if req_type is None:
            if _is_git_url(src_path):
                req_type = 'git'
                req_source = src_path
            elif _is_http_url(src_path):
                req_type = 'url'
                req_source = src_path
            elif _is_file_path(src_path):
                req_type = 'file'
                req_source = src_path
            elif _is_collection_dir(src_path):
                if _is_installed_collection_dir(src_path) and _is_collection_src_dir(src_path):
                    raise AnsibleError(u"Collection requirement at '{path!s}' has both a {manifest_json!s} file and a {galaxy_yml!s}.\nThe requirement must either be an installed collection directory or a source collection directory, not both.".format(path=to_text(src_path, errors='surrogate_or_strict'), manifest_json=to_text(_MANIFEST_JSON), galaxy_yml=to_text(_GALAXY_YAML)))
                req_type = 'dir'
                req_source = src_path
            elif _is_collection_namespace_dir(src_path):
                req_name = None
                req_type = 'subdirs'
                req_source = src_path
            else:
                raise AnsibleError('Failed to automatically detect the collection requirement type.')
        if req_type not in {'file', 'galaxy', 'git', 'url', 'dir', 'subdirs'}:
            raise AnsibleError("The collection requirement entry key 'type' must be one of file, galaxy, git, dir, subdirs, or url.")
        if req_name is None and req_type == 'galaxy':
            raise AnsibleError("Collections requirement entry should contain the key 'name' if it's requested from a Galaxy-like index server.")
        if req_type != 'galaxy' and req_source is None:
            req_source, req_name = (req_name, None)
        if req_type == 'galaxy' and isinstance(req_source, GalaxyAPI) and (not _is_http_url(req_source.api_server)):
            raise AnsibleError("Collections requirement 'source' entry should contain a valid Galaxy API URL but it does not: {not_url!s} is not an HTTP URL.".format(not_url=req_source.api_server))
        if req_type == 'dir' and req_source.endswith(os.path.sep):
            req_source = req_source.rstrip(os.path.sep)
        tmp_inst_req = cls(req_name, req_version, req_source, req_type, req_signature_sources)
        if req_type not in {'galaxy', 'subdirs'} and req_name is None:
            req_name = art_mgr.get_direct_collection_fqcn(tmp_inst_req)
        if req_type not in {'galaxy', 'subdirs'} and req_version == '*':
            req_version = art_mgr.get_direct_collection_version(tmp_inst_req)
        return cls(req_name, req_version, req_source, req_type, req_signature_sources)

    def __repr__(self):
        return '<{self!s} of type {coll_type!r} from {src!s}>'.format(self=self, coll_type=self.type, src=self.src or 'Galaxy')

    def __str__(self):
        return to_native(self.__unicode__())

    def __unicode__(self):
        if self.fqcn is None:
            return u'"virtual collection Git repo"' if self.is_scm else u'"virtual collection namespace"'
        return u'{fqcn!s}:{ver!s}'.format(fqcn=to_text(self.fqcn), ver=to_text(self.ver))

    @property
    def may_have_offline_galaxy_info(self):
        if self.fqcn is None:
            return False
        elif not self.is_dir or self.src is None or (not _is_collection_dir(self.src)):
            return False
        return True

    def construct_galaxy_info_path(self, b_collection_path):
        if not self.may_have_offline_galaxy_info and (not self.type == 'galaxy'):
            raise TypeError('Only installed collections from a Galaxy server have offline Galaxy info')
        b_src = to_bytes(b_collection_path, errors='surrogate_or_strict')
        b_path_parts = b_src.split(to_bytes(os.path.sep))[0:-2]
        b_metadata_dir = to_bytes(os.path.sep).join(b_path_parts)
        b_dir_name = to_bytes(f'{self.namespace}.{self.name}-{self.ver}.info', errors='surrogate_or_strict')
        return os.path.join(b_metadata_dir, b_dir_name, _SOURCE_METADATA_FILE)

    def _get_separate_ns_n_name(self):
        return self.fqcn.split('.')

    @property
    def namespace(self):
        if self.is_virtual:
            raise TypeError('Virtual collections do not have a namespace')
        return self._get_separate_ns_n_name()[0]

    @property
    def name(self):
        if self.is_virtual:
            raise TypeError('Virtual collections do not have a name')
        return self._get_separate_ns_n_name()[-1]

    @property
    def canonical_package_id(self):
        if not self.is_virtual:
            return to_native(self.fqcn)
        return '<virtual namespace from {src!s} of type {src_type!s}>'.format(src=to_native(self.src), src_type=to_native(self.type))

    @property
    def is_virtual(self):
        return self.is_scm or self.is_subdirs

    @property
    def is_file(self):
        return self.type == 'file'

    @property
    def is_dir(self):
        return self.type == 'dir'

    @property
    def namespace_collection_paths(self):
        return [to_native(path) for path in _find_collections_in_subdirs(self.src)]

    @property
    def is_subdirs(self):
        return self.type == 'subdirs'

    @property
    def is_url(self):
        return self.type == 'url'

    @property
    def is_scm(self):
        return self.type == 'git'

    @property
    def is_concrete_artifact(self):
        return self.type in {'git', 'url', 'file', 'dir', 'subdirs'}

    @property
    def is_online_index_pointer(self):
        return not self.is_concrete_artifact

    @property
    def is_pinned(self):
        """Indicate if the version set is considered pinned.

        This essentially computes whether the version field of the current
        requirement explicitly requests a specific version and not an allowed
        version range.

        It is then used to help the resolvelib-based dependency resolver judge
        whether it's acceptable to consider a pre-release candidate version
        despite pre-release installs not being requested by the end-user
        explicitly.

        See https://github.com/ansible/ansible/pull/81606 for extra context.
        """
        version_string = self.ver[0]
        return version_string.isdigit() or not (version_string == '*' or version_string.startswith(('<', '>', '!=')))

    @property
    def source_info(self):
        return self._source_info