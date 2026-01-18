import base64
import copy
import logging
import os
import re
import uuid
import warnings
from io import BytesIO
import botocore
import botocore.auth
from botocore import utils
from botocore.compat import (
from botocore.docs.utils import (
from botocore.endpoint_provider import VALID_HOST_LABEL_RE
from botocore.exceptions import (
from botocore.regions import EndpointResolverBuiltins
from botocore.signers import (
from botocore.utils import (
from botocore import retryhandler  # noqa
from botocore import translate  # noqa
from botocore.compat import MD5_AVAILABLE  # noqa
from botocore.exceptions import MissingServiceIdError  # noqa
from botocore.utils import hyphenize_service_id  # noqa
from botocore.utils import is_global_accesspoint  # noqa
from botocore.utils import SERVICE_NAME_ALIASES  # noqa
class ParameterAlias:

    def __init__(self, original_name, alias_name):
        self._original_name = original_name
        self._alias_name = alias_name

    def alias_parameter_in_call(self, params, model, **kwargs):
        if model.input_shape:
            if self._original_name in model.input_shape.members:
                if self._alias_name in params:
                    if self._original_name in params:
                        raise AliasConflictParameterError(original=self._original_name, alias=self._alias_name, operation=model.name)
                    params[self._original_name] = params.pop(self._alias_name)

    def alias_parameter_in_documentation(self, event_name, section, **kwargs):
        if event_name.startswith('docs.request-params'):
            if self._original_name not in section.available_sections:
                return
            param_section = section.get_section(self._original_name)
            param_type_section = param_section.get_section('param-type')
            self._replace_content(param_type_section)
            param_name_section = param_section.get_section('param-name')
            self._replace_content(param_name_section)
        elif event_name.startswith('docs.request-example'):
            section = section.get_section('structure-value')
            if self._original_name not in section.available_sections:
                return
            param_section = section.get_section(self._original_name)
            self._replace_content(param_section)

    def _replace_content(self, section):
        content = section.getvalue().decode('utf-8')
        updated_content = content.replace(self._original_name, self._alias_name)
        section.clear_text()
        section.write(updated_content)