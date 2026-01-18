import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
class TemplateResourceInfo(ResourceInfo):
    """Store the info needed to start a TemplateResource."""
    description = 'Template'
    __slots__ = ('template_name',)

    def __init__(self, registry, path, value):
        super(TemplateResourceInfo, self).__init__(registry, path, value)
        if self.name.endswith(('.yaml', '.template')):
            self.template_name = self.name
        else:
            self.template_name = value
        self.value = self.template_name

    def get_class(self, files=None):
        from heat.engine.resources import template_resource
        if files and self.template_name in files:
            data = files[self.template_name]
        else:
            if self.user_resource:
                allowed_schemes = template_resource.REMOTE_SCHEMES
            else:
                allowed_schemes = template_resource.LOCAL_SCHEMES
            data = template_resource.TemplateResource.get_template_file(self.template_name, allowed_schemes)
        param_defaults = self.registry.param_defaults
        return template_resource.generate_class_from_template(str(self.name), data, param_defaults)

    def get_class_to_instantiate(self):
        from heat.engine.resources import template_resource
        return template_resource.TemplateResource