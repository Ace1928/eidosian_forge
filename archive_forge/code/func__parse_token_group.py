import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_token_group(self, token, has_baseline, final_targets, extra_flags):
    """validate group token"""
    if len(token) <= 1 or token[-1:] == token[0]:
        self.dist_fatal("'#' must stuck in the begin of group name")
    token = token[1:]
    ghas_baseline, gtargets, gextra_flags = self.parse_target_groups.get(token, (False, None, []))
    if gtargets is None:
        self.dist_fatal("'%s' is an invalid target group name, " % token + 'available target groups are', self.parse_target_groups.keys())
    if ghas_baseline:
        has_baseline = True
    final_targets += [f for f in gtargets if f not in final_targets]
    extra_flags += [f for f in gextra_flags if f not in extra_flags]
    return (has_baseline, final_targets, extra_flags)