from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine.hot import template
from heat.engine import output
from heat.engine import properties
from heat.engine.resources.aws.autoscaling import autoscaling_group as aws_asg
from heat.engine import rsrc_defn
from heat.engine import support
class HOTInterpreter(template.HOTemplate20150430):

    def __new__(cls):
        return object.__new__(cls)

    def __init__(self):
        version = {'heat_template_version': '2015-04-30'}
        super(HOTInterpreter, self).__init__(version)

    def parse(self, stack, snippet, path=''):
        return snippet

    def parse_conditions(self, stack, snippet, path=''):
        return snippet