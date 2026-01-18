import copy
from unittest import mock
from oslo_i18n import fixture as i18n_fixture
import suds
from oslo_vmware import exceptions
from oslo_vmware.tests import base
from oslo_vmware import vim
def _vim_create(self):
    with mock.patch.object(vim.Vim, '__getattr__', self._mock_getattr):
        return vim.Vim()