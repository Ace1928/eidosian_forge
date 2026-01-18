import collections
import copy
import datetime
import functools
import itertools
import os
import pydoc
import signal
import socket
import sys
import eventlet
from oslo_config import cfg
from oslo_context import context as oslo_context
from oslo_log import log as logging
import oslo_messaging as messaging
from oslo_serialization import jsonutils
from oslo_service import service
from oslo_service import threadgroup
from oslo_utils import timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
import webob
from heat.common import context
from heat.common import environment_format as env_fmt
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import messaging as rpc_messaging
from heat.common import policy
from heat.common import service_utils
from heat.engine import api
from heat.engine import attributes
from heat.engine.cfn import template as cfntemplate
from heat.engine import clients
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine import parameter_groups
from heat.engine import properties
from heat.engine import resources
from heat.engine import service_software_config
from heat.engine import stack as parser
from heat.engine import stack_lock
from heat.engine import stk_defn
from heat.engine import support
from heat.engine import template as templatem
from heat.engine import template_files
from heat.engine import update
from heat.engine import worker
from heat.objects import event as event_object
from heat.objects import resource as resource_objects
from heat.objects import service as service_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_api as rpc_worker_api
def _force_exit(*args):
    LOG.info('Graceful exit timeout exceeded, forcing exit.')
    os._exit(-1)