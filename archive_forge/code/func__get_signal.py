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
def _get_signal(self, signal_type=SIGNAL, multiple_signals=False):
    """Return a dictionary with signal details.

        Subclasses can invoke this method to retrieve information of the
        resource signal for the specified transport.
        """
    signal = None
    if self._signal_transport_cfn():
        signal = {'alarm_url': self._get_ec2_signed_url(signal_type=signal_type)}
    elif self._signal_transport_heat():
        signal = self._get_heat_signal_credentials()
        signal['alarm_url'] = self._get_heat_signal_url(project_id=self.stack.stack_user_project_id)
    elif self._signal_transport_temp_url():
        signal = {'alarm_url': self._get_swift_signal_url(multiple_signals=multiple_signals)}
    elif self._signal_transport_zaqar():
        signal = self._get_heat_signal_credentials()
        signal['queue_id'] = self._get_zaqar_signal_queue_id()
    elif self._signal_transport_none():
        signal = {}
    return signal