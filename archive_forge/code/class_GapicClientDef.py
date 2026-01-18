from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class GapicClientDef(object):
    """Struct for info required to instantiate clients/messages for API versions.

  Attributes:
    class_path: str, Path to the package containing api related modules.
    client_full_classpath: str, Full path to the client class for an API
      version.
    async_client_full_classpath: str, Full path to the async client class for an
      API version.
    rest_client_full_classpath: str, Full path to the rest client class for an
      API version.
  """

    def __init__(self, class_path):
        self.class_path = class_path

    @property
    def client_full_classpath(self):
        return self.class_path + '.client.GapicWrapperClient'

    @property
    def async_client_full_classpath(self):
        return self.class_path + '.async_client.GapicWrapperClient'

    @property
    def rest_client_full_classpath(self):
        return self.class_path + '.rest_client.GapicWrapperClient'

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_init_source(self):
        src_fmt = 'GapicClientDef("{0}")'
        return src_fmt.format(self.class_path)

    def __repr__(self):
        return self.get_init_source()