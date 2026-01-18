import configparser
import os
import time
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
def _get_property_from_output(self, output):
    """Create a dictionary from an output

        :param output: the output of the cmd
        """
    obj = {}
    items = self.parser.listing(output)
    for item in items:
        obj[item['Property']] = str(item['Value'])
    return obj