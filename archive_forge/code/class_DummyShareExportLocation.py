import collections
import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share as mshare
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class DummyShareExportLocation(object):

    def __init__(self):
        self.export_location = {'path': 'el'}

    def to_dict(self):
        return self.export_location