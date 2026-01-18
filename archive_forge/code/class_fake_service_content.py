import copy
from unittest import mock
from oslo_i18n import fixture as i18n_fixture
import suds
from oslo_vmware import exceptions
from oslo_vmware.tests import base
from oslo_vmware import vim
class fake_service_content(object):

    def __init__(self):
        self.ServiceContent = {}
        self.ServiceContent.fake = 'fake'