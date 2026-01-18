import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _get_callback_info(callback):
    """Return list containing callback's module and name.

    If the callback is a bound instance method also return the class name.

    :param callback: Function to call
    :type callback: function
    :returns: List containing parent module, (optional class,) function name
    :rtype: list
    """
    module_name = getattr(callback, '__module__', None)
    func_name = callback.__name__
    if inspect.ismethod(callback):
        class_name = reflection.get_class_name(callback.__self__, fully_qualified=False)
        return [module_name, class_name, func_name]
    else:
        return [module_name, func_name]