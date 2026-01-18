from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
from googlecloudsdk.calliope import arg_parsers
class SemicolonSeparatedPemCertificatesFilesList(arg_parsers.ArgList):
    """Reads PEM certificates from all provided files."""

    def __init__(self, **kwargs):
        """Initialize the parser.

    Args:
      **kwargs: Passed verbatim to ArgList.
    """
        super(SemicolonSeparatedPemCertificatesFilesList, self).__init__(element_type=PemCertificatesFile(), custom_delim_char=';', **kwargs)

    def __call__(self, arg_value):
        value = super(SemicolonSeparatedPemCertificatesFilesList, self).__call__(arg_value)
        return list(itertools.chain.from_iterable(value))