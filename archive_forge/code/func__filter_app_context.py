from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def _filter_app_context(self, object_type, section, name, global_conf, local_conf, global_additions):
    if 'next' not in local_conf:
        raise LookupError("The [%s] section in %s is missing a 'next' setting" % (section, self.filename))
    next_name = local_conf.pop('next')
    context = LoaderContext(None, FILTER_APP, None, global_conf, local_conf, self)
    context.next_context = self.get_context(APP, next_name, global_conf)
    if 'use' in local_conf:
        context.filter_context = self._context_from_use(FILTER, local_conf, global_conf, global_additions, section)
    else:
        context.filter_context = self._context_from_explicit(FILTER, local_conf, global_conf, global_additions, section)
    return context