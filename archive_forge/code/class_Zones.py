from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Zones(base.Group):
    """Manage Access Context Manager service perimeters.

  A service perimeter describes a set of Google Cloud Platform resources which
  can freely import and export data amongst themselves, but not externally.

  Currently, the only allowed members of a service perimeter are projects.
  """