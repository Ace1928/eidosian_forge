import types
import os
import string
import uuid
from paste.deploy import appconfig
from paste.script import copydir
from paste.script.command import Command, BadCommand, run as run_command
from paste.script.util import secret
from paste.util import import_string
import paste.script.templates
import pkg_resources
def _call_setup_app(self, func, command, filename, section, vars):
    filename = os.path.abspath(filename)
    if ':' in section:
        section = section.split(':', 1)[1]
    conf = 'config:%s#%s' % (filename, section)
    conf = appconfig(conf)
    conf.filename = filename
    func(command, conf, vars)