import os
import six
from genshi.template.base import TemplateError
from genshi.util import LRUCache
class TemplateNotFound(TemplateError):
    """Exception raised when a specific template file could not be found."""

    def __init__(self, name, search_path):
        """Create the exception.
        
        :param name: the filename of the template
        :param search_path: the search path used to lookup the template
        """
        TemplateError.__init__(self, 'Template "%s" not found' % name)
        self.search_path = search_path