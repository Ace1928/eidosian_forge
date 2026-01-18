import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_policy_keepsort(self, has_baseline, final_targets, extra_flags):
    """leave a notice that $keep_sort is on"""
    self.dist_log("policy 'keep_sort' is on, dispatch-able targets", final_targets, "\nare 'not' sorted depend on the highest interest butas specified in the dispatch-able source or the extra group")
    return (has_baseline, final_targets, extra_flags)