import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
def NormjoinRulePathForceCMakeSource(base_path, rel_path, rule_source):
    if rel_path.startswith(('${RULE_INPUT_PATH}', '${RULE_INPUT_DIRNAME}')):
        if any([rule_source.startswith(var) for var in FULL_PATH_VARS]):
            return rel_path
    return NormjoinPathForceCMakeSource(base_path, rel_path)