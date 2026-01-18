from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
class PolicyConstraint(constraints.BaseCustomConstraint):
    expected_exceptions = (exceptions.HttpException,)

    def validate_with_client(self, client, value):
        client.client(CLIENT_NAME).get_policy(value)