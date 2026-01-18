from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionAddressesMoveRequest(_messages.Message):
    """A RegionAddressesMoveRequest object.

  Fields:
    description: An optional destination address description if intended to be
      different from the source.
    destinationAddress: The URL of the destination address to move to. This
      can be a full or partial URL. For example, the following are all valid
      URLs to a address: -
      https://www.googleapis.com/compute/v1/projects/project/regions/region
      /addresses/address - projects/project/regions/region/addresses/address
      Note that destination project must be different from the source project.
      So /regions/region/addresses/address is not valid partial url.
  """
    description = _messages.StringField(1)
    destinationAddress = _messages.StringField(2)