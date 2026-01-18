import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import role_assignments
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
class DummyRoleAssignment(generic_resource.GenericResource, MixinClass):
    properties_schema = {}
    properties_schema.update(MixinClass.mixin_properties_schema)

    def validate(self):
        super(DummyRoleAssignment, self).validate()
        self.validate_assignment_properties()