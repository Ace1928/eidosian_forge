from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def _pipeline_app_context(self, object_type, section, name, global_conf, local_conf, global_additions):
    if 'pipeline' not in local_conf:
        raise LookupError("The [%s] section in %s is missing a 'pipeline' setting" % (section, self.filename))
    pipeline = local_conf.pop('pipeline').split()
    if local_conf:
        raise LookupError('The [%s] pipeline section in %s has extra (disallowed) settings: %s' % (section, self.filename, ', '.join(local_conf.keys())))
    context = LoaderContext(None, PIPELINE, None, global_conf, local_conf, self)
    context.app_context = self.get_context(APP, pipeline[-1], global_conf)
    context.filter_contexts = [self.get_context(FILTER, name, global_conf) for name in pipeline[:-1]]
    return context