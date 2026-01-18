import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_policy_autovec(self, has_baseline, final_targets, extra_flags):
    """skip features that has no auto-vectorized support by compiler"""
    skipped = []
    for tar in final_targets[:]:
        if isinstance(tar, str):
            can = self.feature_can_autovec(tar)
        else:
            can = all([self.feature_can_autovec(t) for t in tar])
        if not can:
            final_targets.remove(tar)
            skipped.append(tar)
    if skipped:
        self.dist_log('skip non auto-vectorized features', skipped)
    return (has_baseline, final_targets, extra_flags)