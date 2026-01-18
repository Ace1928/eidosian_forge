import copy
import functools
import itertools
import operator
import os
import threading
import uuid
from oslo_config import cfg
from oslo_log import log
from pycadf import reason
from keystone import assignment  # TODO(lbragstad): Decouple this dependency
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from oslo_utils import timeutils
def _assert_default_project_id_is_not_domain(self, default_project_id):
    if default_project_id:
        try:
            project_ref = PROVIDERS.resource_api.get_project(default_project_id)
            if project_ref['is_domain'] is True:
                msg = _("User's default project ID cannot be a domain ID: %s")
                raise exception.ValidationError(message=msg % default_project_id)
        except exception.ProjectNotFound:
            pass