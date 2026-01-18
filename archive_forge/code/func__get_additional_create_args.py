from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
def _get_additional_create_args(self, version):
    return {'clustering_api_version': version or '1'}