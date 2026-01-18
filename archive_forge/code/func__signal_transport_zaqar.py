from urllib import parse
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine.clients.os import swift
from heat.engine.resources import stack_user
def _signal_transport_zaqar(self):
    return self.properties.get(self.SIGNAL_TRANSPORT) == self.ZAQAR_SIGNAL