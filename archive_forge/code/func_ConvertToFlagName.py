from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ConvertToFlagName(name):
    """Convert name to flag format (e.g. '--foo-bar')."""
    return AddFlagPrefix(name).lower().replace('_', '-').replace(' ', '-')