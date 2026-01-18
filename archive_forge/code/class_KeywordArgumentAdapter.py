import configparser
import logging
import logging.config
import logging.handlers
import os
import platform
import sys
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_utils import units
from oslo_log._i18n import _
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
class KeywordArgumentAdapter(BaseLoggerAdapter):
    """Logger adapter to add keyword arguments to log record's extra data

    Keywords passed to the log call are added to the "extra"
    dictionary passed to the underlying logger so they are emitted
    with the log message and available to the format string.

    Special keywords:

    extra
      An existing dictionary of extra values to be passed to the
      logger. If present, the dictionary is copied and extended.
    resource
      A dictionary-like object containing a ``name`` key or ``type``
       and ``id`` keys.

    """

    def process(self, msg, kwargs):
        extra = {}
        extra.update(self.extra)
        if 'extra' in kwargs:
            extra.update(kwargs.pop('extra'))
        for name in list(kwargs.keys()):
            if name == 'exc_info':
                continue
            extra[name] = kwargs.pop(name)
        extra['extra_keys'] = list(sorted(extra.keys()))
        kwargs['extra'] = extra
        resource = kwargs['extra'].get('resource', None)
        if resource:
            if not resource.get('name', None):
                resource_type = resource.get('type', None)
                resource_id = resource.get('id', None)
                if resource_type and resource_id:
                    kwargs['extra']['resource'] = '[' + resource_type + '-' + resource_id + '] '
            else:
                kwargs['extra']['resource'] = '[' + resource.get('name', '') + '] '
        return (msg, kwargs)