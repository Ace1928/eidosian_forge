import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
@staticmethod
def add_federated_schema_version_option(parser):
    parser.add_argument('--schema-version', metavar='<schema_version>', required=False, default=None, help=_("The federated attribute mapping schema version. The default value on the client side is 'None'; however, that will lead the backend to set the default according to 'attribute_mapping_default_schema_version' option."))