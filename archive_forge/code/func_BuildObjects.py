from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_builder
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def BuildObjects(default_class, stream, loader=yaml.loader.SafeLoader):
    """Build objects from stream.

  Handles the basic case of loading all the objects from a stream.

  Args:
    default_class: Class that is instantiated upon the detection of a new
      document.  An instance of this class will act as the document itself.
    stream: String document or open file object to process as per the
      yaml.parse method.  Any object that implements a 'read()' method which
      returns a string document will work with the YAML parser.
    loader_class: Used for dependency injection.

  Returns:
    List of default_class instances parsed from the stream.
  """
    builder = ObjectBuilder(default_class)
    handler = yaml_builder.BuilderHandler(builder)
    listener = yaml_listener.EventListener(handler)
    listener.Parse(stream, loader, version=(1, 1))
    return handler.GetResults()