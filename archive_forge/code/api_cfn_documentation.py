import eventlet
import __original_module_threading as orig_threading
import threading  # noqa
import sys
from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from oslo_reports import guru_meditation_report as gmr
from oslo_service import systemd
from heat.common import config
from heat.common import messaging
from heat.common import profiler
from heat.common import wsgi
from heat import version
Heat API Server.

This implements an approximation of the Amazon CloudFormation API and
translates it into a native representation.  It then calls the heat-engine via
AMQP RPC to implement them.
