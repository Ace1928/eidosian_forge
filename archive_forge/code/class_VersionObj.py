import threading
from unittest import mock
import greenlet
from oslo_config import cfg
from oslotest import base
from oslo_reports.generators import conf as os_cgen
from oslo_reports.generators import threading as os_tgen
from oslo_reports.generators import version as os_pgen
from oslo_reports.models import threading as os_tmod
class VersionObj(object):

    def product_string(self):
        return 'Sharp Cheddar'

    def version_string_with_package(self):
        return '1.0.0'