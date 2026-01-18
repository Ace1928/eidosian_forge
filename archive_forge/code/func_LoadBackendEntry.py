from __future__ import absolute_import
import os
def LoadBackendEntry(backend_entry):
    """Parses a BackendEntry object from a string.

  Args:
    backend_entry: a backend entry, as a string

  Returns:
    A BackendEntry object.
  """
    builder = yaml_object.ObjectBuilder(BackendEntry)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(backend_entry)
    entries = handler.GetResults()
    if len(entries) < 1:
        raise BadConfig('Empty backend configuration.')
    if len(entries) > 1:
        raise BadConfig('Multiple backend entries were found in configuration.')
    return entries[0].Init()