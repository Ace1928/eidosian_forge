from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class DuplicateConfigError(exceptions.Error):
    """Two config files of the same type."""

    def __init__(self, path1, path2, config_type):
        super(DuplicateConfigError, self).__init__('[{path1}] and [{path2}] are both trying to define a {t} config file. Only one config file of the same type can be updated at once.'.format(path1=path1, path2=path2, t=config_type))