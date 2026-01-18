from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidGoModuleError(exceptions.Error):
    """Raised when the Go module source code cannot be packaged into a go.zip."""