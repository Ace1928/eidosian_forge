from __future__ import absolute_import
import os
class BackendInfoExternal(validation.Validated):
    """BackendInfoExternal describes all backend entries for an application."""
    ATTRIBUTES = {BACKENDS: validation.Optional(validation.Repeated(BackendEntry))}