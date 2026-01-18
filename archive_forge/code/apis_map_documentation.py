from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
Struct for info required to instantiate clients/messages for API versions.

  Attributes:
    class_path: str, Path to the package containing api related modules.
    client_full_classpath: str, Full path to the client class for an API
      version.
    async_client_full_classpath: str, Full path to the async client class for an
      API version.
    rest_client_full_classpath: str, Full path to the rest client class for an
      API version.
  