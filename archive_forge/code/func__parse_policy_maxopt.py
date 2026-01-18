import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_policy_maxopt(self, has_baseline, final_targets, extra_flags):
    """append the compiler optimization flags"""
    if self.cc_has_debug:
        self.dist_log("debug mode is detected, policy 'maxopt' is skipped.")
    elif self.cc_noopt:
        self.dist_log("optimization is disabled, policy 'maxopt' is skipped.")
    else:
        flags = self.cc_flags['opt']
        if not flags:
            self.dist_log("current compiler doesn't support optimization flags, policy 'maxopt' is skipped", stderr=True)
        else:
            extra_flags += flags
    return (has_baseline, final_targets, extra_flags)