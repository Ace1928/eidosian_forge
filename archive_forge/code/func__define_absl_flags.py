from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from absl import flags
def _define_absl_flags(self, absl_flags):
    """Defines flags from absl_flags."""
    key_flags = set(absl_flags.get_key_flags_for_module(sys.argv[0]))
    for name in absl_flags:
        if name in _BUILT_IN_FLAGS:
            continue
        flag_instance = absl_flags[name]
        if name == flag_instance.name:
            suppress = flag_instance not in key_flags
            self._define_absl_flag(flag_instance, suppress)