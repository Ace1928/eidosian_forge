from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddUpdateMaskFlag(parser):
    """Adds a --update-mask flag to the given parser."""
    help_text = 'Used to specify the fields to be overwritten in the stream resource by the update.\n  If the update mask is used, then a field will be overwritten only if it is in the mask. If the user does not provide a mask then all fields will be overwritten.\n  This is a comma-separated list of fully qualified names of fields, written as snake_case or camelCase. Example: "display_name, source_config.oracle_source_config".'
    parser.add_argument('--update-mask', help=help_text)