from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class AccessibilityHelp(base.TopicCommand):
    """Reference for `Accessibility` features.

  The `accessibility/screen_reader` property when set to true will change some
  behavior to make gcloud more screen reader friendly. Currently the following
  changes are implemented:

  * For progress trackers, instead of unicode spinners, the phrase 'working'
    is displayed on stderr, every second while gcloud is working.
  * For progress bars, progress is displayed as a percentage, outputted to
    stderr.
  * For boxed tables, instead of the queried resources being displayed in tables
    drawn in Unicode, results are rendered as a flattened list of items.
    Also consider using the --format flag to define your own format.

  To turn this on, run:

    $ gcloud config set accessibility/screen_reader true

  Accessibility support is still in early stages. Please report any issues that
  you would like fixed using `gcloud feedback`.
  """