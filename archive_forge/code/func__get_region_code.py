from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer.appliances import flags
def _get_region_code(order_resource, args):
    """Get region code either from the country arg or the previous value."""
    if hasattr(args, 'country'):
        return args.country
    return order_resource.address.regionCode