import os
import six
from genshi.template.base import TemplateError
from genshi.util import LRUCache
def _instantiate(self, cls, fileobj, filepath, filename, encoding=None):
    """Instantiate and return the `Template` object based on the given
        class and parameters.
        
        This function is intended for subclasses to override if they need to
        implement special template instantiation logic. Code that just uses
        the `TemplateLoader` should use the `load` method instead.
        
        :param cls: the class of the template object to instantiate
        :param fileobj: a readable file-like object containing the template
                        source
        :param filepath: the absolute path to the template file
        :param filename: the path to the template file relative to the search
                         path
        :param encoding: the encoding of the template to load; defaults to the
                         ``default_encoding`` of the loader instance
        :return: the loaded `Template` instance
        :rtype: `Template`
        """
    if encoding is None:
        encoding = self.default_encoding
    return cls(fileobj, filepath=filepath, filename=filename, loader=self, encoding=encoding, lookup=self.variable_lookup, allow_exec=self.allow_exec)