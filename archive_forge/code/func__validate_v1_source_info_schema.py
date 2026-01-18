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
def _validate_v1_source_info_schema(namespace, name, version, provided_arguments):
    argument_spec_data = dict(format_version=dict(choices=['1.0.0']), download_url=dict(), version_url=dict(), server=dict(), signatures=dict(type=list, suboptions=dict(signature=dict(), pubkey_fingerprint=dict(), signing_service=dict(), pulp_created=dict())), name=dict(choices=[name]), namespace=dict(choices=[namespace]), version=dict(choices=[version]))
    if not isinstance(provided_arguments, dict):
        raise AnsibleError(f'Invalid offline source info for {namespace}.{name}:{version}, expected a dict and got {type(provided_arguments)}')
    validator = ArgumentSpecValidator(argument_spec_data)
    validation_result = validator.validate(provided_arguments)
    return validation_result.error_messages