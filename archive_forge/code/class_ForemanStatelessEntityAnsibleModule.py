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
class ForemanStatelessEntityAnsibleModule(ForemanAnsibleModule):
    """ Base class for Foreman entities without a state. To use it, subclass it with the following convention:
        To manage my_entity entity, create the following sub class::

            class ForemanMyEntityModule(ForemanStatelessEntityAnsibleModule):
                pass

        and use that class to instantiate module::

            module = ForemanMyEntityModule(
                argument_spec=dict(
                    [...]
                ),
                foreman_spec=dict(
                    [...]
                ),
            )

        It adds the following attributes:

        * entity_key (str): field used to search current entity. Defaults to value provided by `ENTITY_KEYS` or 'name' if no value found.
        * entity_name (str): name of the current entity.
          By default deduce the entity name from the class name (eg: 'ForemanProvisioningTemplateModule' class will produce 'provisioning_template').
        * entity_opts (dict): Dict of options for base entity. Same options can be provided for subentities described in foreman_spec.

        The main entity is referenced with the key `entity` in the `foreman_spec`.
    """

    def __init__(self, **kwargs):
        self.entity_key = kwargs.pop('entity_key', 'name')
        self.entity_name = kwargs.pop('entity_name', self.entity_name_from_class)
        entity_opts = kwargs.pop('entity_opts', {})
        super(ForemanStatelessEntityAnsibleModule, self).__init__(**kwargs)
        if 'resource_type' not in entity_opts:
            entity_opts['resource_type'] = inflector.pluralize(self.entity_name)
        if 'thin' not in entity_opts:
            entity_opts['thin'] = None
        if 'failsafe' not in entity_opts:
            entity_opts['failsafe'] = True
        if 'search_operator' not in entity_opts:
            entity_opts['search_operator'] = '='
        if 'search_by' not in entity_opts:
            entity_opts['search_by'] = ENTITY_KEYS.get(entity_opts['resource_type'], 'name')
        self.foreman_spec.update(_foreman_spec_helper(dict(entity=dict(type='entity', flat_name='id', ensure=False, **entity_opts)))[0])
        if 'parent' in self.foreman_spec and self.foreman_spec['parent'].get('type') == 'entity':
            if 'resouce_type' not in self.foreman_spec['parent']:
                self.foreman_spec['parent']['resource_type'] = self.foreman_spec['entity']['resource_type']
            if 'failsafe' not in self.foreman_spec['parent']:
                self.foreman_spec['parent']['failsafe'] = True
            current, parent = split_fqn(self.foreman_params[self.entity_key])
            if isinstance(self.foreman_params.get('parent'), six.string_types):
                if parent:
                    self.fail_json(msg='Please specify the parent either separately, or as part of the title.')
                parent = self.foreman_params['parent']
            elif parent:
                self.foreman_params['parent'] = parent
            self.foreman_params[self.entity_key] = current
            self.foreman_params['entity'] = build_fqn(current, parent)
        else:
            self.foreman_params['entity'] = self.foreman_params.get(self.entity_key)

    @property
    def entity_name_from_class(self):
        """
        The entity name derived from the class name.

        The class name must follow the following name convention:

        * It starts with ``Foreman`` or ``Katello``.
        * It ends with ``Module``.

        This will convert the class name ``ForemanMyEntityModule`` to the entity name ``my_entity``.

        Examples:

        * ``ForemanArchitectureModule`` => ``architecture``
        * ``ForemanProvisioningTemplateModule`` => ``provisioning_template``
        * ``KatelloProductMudule`` => ``product``
        """
        class_name = re.sub('(?<=[a-z])[A-Z]|[A-Z](?=[^A-Z])', '_\\g<0>', self.__class__.__name__).lower().strip('_')
        return '_'.join(class_name.split('_')[1:-1])