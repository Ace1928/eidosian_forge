from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ..module_utils.gcp_utils import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def _get_query_options(self, filters):
    """
        :param config_data: contents of the inventory config file
        :return A fully built query string
        """
    if not filters:
        return ''
    if len(filters) == 1:
        return filters[0]
    else:
        queries = []
        for f in filters:
            if f[0] != '(' and f[-1] != ')':
                queries.append('(%s)' % ''.join(f))
            else:
                queries.append(f)
        return ' '.join(queries)