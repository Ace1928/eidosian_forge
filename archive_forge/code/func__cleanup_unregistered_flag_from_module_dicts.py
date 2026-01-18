from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def _cleanup_unregistered_flag_from_module_dicts(self, flag_obj):
    """Cleans up unregistered flags from all module -> [flags] dictionaries.

    If flag_obj is registered under either its long name or short name, it
    won't be removed from the dictionaries.

    Args:
      flag_obj: Flag, the Flag instance to clean up for.
    """
    if self._flag_is_registered(flag_obj):
        return
    for flags_by_module_dict in (self.flags_by_module_dict(), self.flags_by_module_id_dict(), self.key_flags_by_module_dict()):
        for flags_in_module in six.itervalues(flags_by_module_dict):
            while flag_obj in flags_in_module:
                flags_in_module.remove(flag_obj)