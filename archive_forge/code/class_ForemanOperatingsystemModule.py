from __future__ import absolute_import, division, print_function
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import (
class ForemanOperatingsystemModule(ParametersMixin, ForemanEntityAnsibleModule):
    PARAMETERS_FLAT_NAME = 'os_parameters_attributes'