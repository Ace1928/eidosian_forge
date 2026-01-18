from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import iteritems
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import (

        Return the list of differences between the current parameters and the existing volume.

        :return: list of options that differ
        