import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_policy_werror(self, has_baseline, final_targets, extra_flags):
    """force warnings to treated as errors"""
    flags = self.cc_flags['werror']
    if not flags:
        self.dist_log("current compiler doesn't support werror flags, warnings will 'not' treated as errors", stderr=True)
    else:
        self.dist_log('compiler warnings are treated as errors')
        extra_flags += flags
    return (has_baseline, final_targets, extra_flags)