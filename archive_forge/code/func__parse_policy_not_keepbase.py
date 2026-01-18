import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_policy_not_keepbase(self, has_baseline, final_targets, extra_flags):
    """skip all baseline features"""
    skipped = []
    for tar in final_targets[:]:
        is_base = False
        if isinstance(tar, str):
            is_base = tar in self.parse_baseline_names
        else:
            is_base = all([f in self.parse_baseline_names for f in tar])
        if is_base:
            skipped.append(tar)
            final_targets.remove(tar)
    if skipped:
        self.dist_log('skip baseline features', skipped)
    return (has_baseline, final_targets, extra_flags)