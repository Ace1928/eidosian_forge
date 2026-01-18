import sys
import traceback
from oslo_config import cfg
from oslo_utils import reflection
import webob
from heat.common import exception
from heat.common import serializers
from heat.common import wsgi
Replace error body with something the client can parse.