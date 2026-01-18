import json
import os
import betamax.serializers.base
import yaml
def _should_use_block(value):
    for c in u'\n\r\x1c\x1d\x1e\x85\u2028\u2029':
        if c in value:
            return True
    return False