from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import six
class RequiredError(DetailedArgumentError):
    """Required error."""

    def __init__(self, **kwargs):
        super(RequiredError, self).__init__('Must be specified.', **kwargs)