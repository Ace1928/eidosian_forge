from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def _filter_out_none_entries_from_list(self, alist, allow_empty_list_or_dict):
    """take a list as input and return a list without elements whose values are None
           return empty dicts or lists if allow_empty_list_or_dict otherwise skip empty dicts or lists.
        """
    result = []
    for item in alist:
        if isinstance(item, (list, dict)):
            sub = self.filter_out_none_entries(item, allow_empty_list_or_dict)
            if sub or allow_empty_list_or_dict:
                result.append(sub)
        elif item is not None:
            result.append(item)
    return result