import collections
import gc
import io
import ipaddress
import json
import platform
import socket
import sys
import traceback
from debtcollector import removals
import jinja2
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import stevedore
import webob.dec
import webob.exc
import webob.response
from oslo_middleware import base
from oslo_middleware.healthcheck import opts
def _make_head_response(self, results, healthy):
    return ('', 'text/plain')