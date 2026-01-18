from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetEnvVarsAsDict(env_vars):
    if env_vars:
        return {prop.key: prop.value for prop in env_vars.additionalProperties}
    else:
        return {}