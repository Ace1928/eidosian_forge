from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os.path
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _MakeApiDict(message_module, collection_dict):
    """Returns a dictionary of API attributes from its messages module.

  Args:
    message_module: the messages module for the API (default version)
    collection_dict: a dictionary containing collection info from registry
  """
    api_dict = {}
    try:
        resource_message = getattr(message_module, _GetResourceMessageClassName(collection_dict['singular_name']))
        args = [field.__dict__['name'] for field in resource_message.all_fields() if field.__dict__['name'] != 'name']
        api_dict['create_args'] = {arg: '-'.join([w.lower() for w in re.findall('^[a-z]*|[A-Z][a-z]*', arg)]) for arg in args}
    except AttributeError:
        api_dict['create_args'] = {}
        log.status.Print('Cannot find ' + _GetResourceMessageClassName(collection_dict['singular_name']) + ' in message module.')
    return api_dict