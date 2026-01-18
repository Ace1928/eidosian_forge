from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def _context_from_use(self, object_type, local_conf, global_conf, global_additions, section):
    use = local_conf.pop('use')
    context = self.get_context(object_type, name=use, global_conf=global_conf)
    context.global_conf.update(global_additions)
    context.local_conf.update(local_conf)
    if '__file__' in global_conf:
        context.global_conf['__file__'] = global_conf['__file__']
    context.loader = self
    if context.protocol is None:
        section_protocol = section.split(':', 1)[0]
        if section_protocol in ('application', 'app'):
            context.protocol = 'paste.app_factory'
        elif section_protocol in ('composit', 'composite'):
            context.protocol = 'paste.composit_factory'
        else:
            context.protocol = 'paste.%s_factory' % section_protocol
    return context