from __future__ import absolute_import
import os
def LoadBackendInfo(backend_info, open_fn=None):
    """Parses a BackendInfoExternal object from a string.

  Args:
    backend_info: a backends stanza (list of backends) as a string
    open_fn: Function for opening files. Unused.

  Returns:
    A BackendInfoExternal object.
  """
    builder = yaml_object.ObjectBuilder(BackendInfoExternal)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(backend_info)
    backend_info = handler.GetResults()
    if len(backend_info) < 1:
        return BackendInfoExternal(backends=[])
    if len(backend_info) > 1:
        raise BadConfig("Only one 'backends' clause is allowed.")
    info = backend_info[0]
    if not info.backends:
        return BackendInfoExternal(backends=[])
    for backend in info.backends:
        backend.Init()
    return info