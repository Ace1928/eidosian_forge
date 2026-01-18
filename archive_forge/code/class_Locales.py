from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Locales(base.Group):
    """Explore iOS locales available for testing."""
    detailed_help = {'EXAMPLES': '          To list all available iOS locales which can be used for testing\n          international applications, run:\n\n            $ {command} list\n\n          To view information about a specific locale, run:\n\n            $ {command} describe LOCALE\n          '}

    @staticmethod
    def Args(parser):
        """Method called by Calliope to register flags common to this sub-group.

    Args:
      parser: An argparse parser used to add arguments that immediately follow
          this group in the CLI. Positional arguments are allowed.
    """