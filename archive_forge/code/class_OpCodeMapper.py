import json
import os
import re
import sys
import numpy as np
class OpCodeMapper:
    """Maps an opcode index to an op name."""

    def __init__(self, data):
        self.code_to_name = {}
        for idx, d in enumerate(data['operator_codes']):
            self.code_to_name[idx] = BuiltinCodeToName(d['builtin_code'])
            if self.code_to_name[idx] == 'CUSTOM':
                self.code_to_name[idx] = NameListToString(d['custom_code'])

    def __call__(self, x):
        if x not in self.code_to_name:
            s = '<UNKNOWN>'
        else:
            s = self.code_to_name[x]
        return '%s (%d)' % (s, x)