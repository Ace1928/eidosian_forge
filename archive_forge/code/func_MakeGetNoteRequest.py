from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
from six.moves import range
def MakeGetNoteRequest(note_name, default_project):
    client = apis.GetClientInstance('containeranalysis', 'v1alpha1')
    messages = apis.GetMessagesModule('containeranalysis', 'v1alpha1')
    note_ref = _GetNoteRef(note_name, default_project)
    request = messages.ContaineranalysisProvidersNotesGetRequest(name=note_ref.RelativeName())
    return client.providers_notes.Get(request)