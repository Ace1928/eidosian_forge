from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
from six.moves import range
def _GetNoteRef(note_name, default_project):
    try:
        return resources.REGISTRY.ParseRelativeName(note_name, 'containeranalysis.providers.notes')
    except resources.InvalidResourceException:
        return resources.REGISTRY.Parse(note_name, params={'providersId': default_project}, collection='containeranalysis.providers.notes')