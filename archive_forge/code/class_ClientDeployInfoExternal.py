from __future__ import absolute_import
import os
class ClientDeployInfoExternal(validation.Validated):
    """Describes the format of a client_deployinfo.yaml file."""
    ATTRIBUTES = {RUNTIME: appinfo.RUNTIME_RE_STRING, START_TIME_USEC: validation.TYPE_LONG, END_TIME_USEC: validation.TYPE_LONG, REQUESTS: validation.Optional(validation.Repeated(Request)), SUCCESS: validation.TYPE_BOOL, SDK_VERSION: validation.Optional(validation.TYPE_STR)}