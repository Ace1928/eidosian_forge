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
def _set_list_limit_in_hints(self, hints, driver):
    """Set list limit in hints from driver.

        If a hints list is provided, the wrapper will insert the relevant
        limit into the hints so that the underlying driver call can try and
        honor it. If the driver does truncate the response, it will update the
        'truncated' attribute in the 'limit' entry in the hints list, which
        enables the caller of this function to know if truncation has taken
        place. If, however, the driver layer is unable to perform truncation,
        the 'limit' entry is simply left in the hints list for the caller to
        handle.

        A _get_list_limit() method is required to be present in the object
        class hierarchy, which returns the limit for this backend to which
        we will truncate.

        If a hints list is not provided in the arguments of the wrapped call
        then any limits set in the config file are ignored.  This allows
        internal use of such wrapped methods where the entire data set is
        needed as input for the calculations of some other API (e.g. get role
        assignments for a given project).

        This method, specific to identity manager, is used instead of more
        general response_truncated, because the limit for identity entities
        can be overridden in domain-specific config files. The driver to use
        is determined during processing of the passed parameters and
        response_truncated is designed to set the limit before any processing.
        """
    if hints is None:
        return
    list_limit = driver._get_list_limit()
    if list_limit:
        hints.set_limit(list_limit)