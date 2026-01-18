import logging
import sys
from unittest import mock
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
from oslotest import base as test_base
from oslo_log import formatters
from oslo_log import log
def _fake_context():
    ctxt = context.RequestContext(user_id='user', project_id='tenant', project_domain_id='pdomain', user_domain_id='udomain', overwrite=True)
    return ctxt