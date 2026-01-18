from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
class StoreGlobalAction(argparse._StoreConstAction):
    """Return "global" if the --global argument is used."""

    def __init__(self, option_strings, dest, default='', required=False, help=None):
        super(StoreGlobalAction, self).__init__(option_strings=option_strings, dest=dest, const='global', default=default, required=required, help=help)