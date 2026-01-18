from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
class ParametersMixin(ParametersMixinBase):
    """
    Parameters Mixin to extend a :class:`ForemanAnsibleModule` (or any subclass) to work with entities that support parameters.

    This allows to submit parameters to Foreman in the same request as modifying the main entity, thus making the parameters
    available to any action that might be triggered when the entity is saved.

    By default, parametes are submited to the API using the ``<entity_name>_parameters_attributes`` key.
    If you need to override this, set the ``PARAMETERS_FLAT_NAME`` attribute to the key that shall be used instead.

    This adds optional ``parameters`` parameter to the module. It also enhances the ``run()`` method to properly handle the
    provided parameters.
    """

    def __init__(self, **kwargs):
        self.entity_name = kwargs.pop('entity_name', self.entity_name_from_class)
        parameters_flat_name = getattr(self, 'PARAMETERS_FLAT_NAME', None) or '{0}_parameters_attributes'.format(self.entity_name)
        foreman_spec = dict(parameters=dict(type='list', elements='dict', options=parameter_ansible_spec, flat_name=parameters_flat_name))
        foreman_spec.update(kwargs.pop('foreman_spec', {}))
        super(ParametersMixin, self).__init__(foreman_spec=foreman_spec, **kwargs)
        self.validate_parameters()

    def run(self, **kwargs):
        entity = self.lookup_entity('entity')
        if not self.desired_absent:
            if entity and 'parameters' in entity:
                entity['parameters'] = parameters_list_to_str_list(entity['parameters'])
            parameters = self.foreman_params.get('parameters')
            if parameters is not None:
                self.foreman_params['parameters'] = parameters_list_to_str_list(parameters)
        return super(ParametersMixin, self).run(**kwargs)