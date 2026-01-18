from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetStatusForRepo(obj):
    """Get the status for a repo.

  Args:
    obj: The RepoSync|RootSync object.

  Returns:
    a SingleRepoStatus object that represents the RepoSync|RootSync object.
  """
    stalled = _GetConditionForType(obj, 'Stalled')
    if stalled and stalled['status'] == 'True':
        return SingleRepoStatus('STALLED', [stalled['message']], '')
    reconciling = _GetConditionForType(obj, 'Reconciling')
    if reconciling and reconciling['status'] == 'True':
        return SingleRepoStatus('RECONCILING', [], '')
    syncing = _GetConditionForType(obj, 'Syncing')
    if syncing:
        error_source_refs = _GetPathValue(syncing, ['errorSourceRefs'], [])
        errs = _GetErrorFromSourceRef(obj, error_source_refs)
        errs.extend(_GetPathValue(syncing, ['errors'], []))
        commit = _GetPathValue(syncing, ['commit'], '')
        if errs:
            return SingleRepoStatus('ERROR', _GetErrorMessages(errs), commit)
        if syncing['status'] == 'True':
            return SingleRepoStatus('PENDING', [], commit)
        return SingleRepoStatus('SYNCED', [], commit)
    rendering = _GetPathValue(obj, ['status', 'rendering', 'commit'], '')
    source = _GetPathValue(obj, ['status', 'source', 'commit'], '')
    sync = _GetPathValue(obj, ['status', 'sync', 'commit'], '')
    status = ''
    if not rendering:
        errors = []
        if not source and (not sync):
            status = 'PENDING'
        elif source != sync:
            errors = _GetPathValue(obj, ['status', 'source', 'errors'], [])
            if errors:
                status = 'ERROR'
            else:
                status = 'PENDING'
        else:
            errors += _GetPathValue(obj, ['status', 'source', 'errors'], [])
            errors += _GetPathValue(obj, ['status', 'sync', 'errors'], [])
            if errors:
                status = 'ERROR'
            else:
                status = 'SYNCED'
        return SingleRepoStatus(status, _GetErrorMessages(errors), source)
    stalled_ts = _GetPathValue(stalled, ['lastUpdateTime'], '2000-01-01T23:50:20Z')
    reconciling_ts = _GetPathValue(reconciling, ['lastUpdateTime'], '2000-01-01T23:50:20Z')
    rendering_ts = _GetPathValue(obj, ['status', 'rendering', 'lastUpdate'], '2000-01-01T23:50:20Z')
    source_ts = _GetPathValue(obj, ['status', 'source', 'lastUpdate'], '2000-01-01T23:50:20Z')
    sync_ts = _GetPathValue(obj, ['status', 'sync', 'lastUpdate'], '2000-01-01T23:50:20Z')
    stalled_time = _TimeFromString(stalled_ts)
    reconciling_time = _TimeFromString(reconciling_ts)
    rendering_time = _TimeFromString(rendering_ts)
    source_time = _TimeFromString(source_ts)
    sync_time = _TimeFromString(sync_ts)
    if stalled_time > rendering_time and stalled_time > source_time and (stalled_time > sync_time) or (reconciling_time > rendering_time and reconciling_time > source_time and (stalled_time > sync_time)):
        return SingleRepoStatus('PENDING', [], '')
    if rendering_time > source_time and rendering_time > sync_time:
        errors = _GetPathValue(obj, ['status', 'rendering', 'errors'], [])
        if errors:
            status = 'ERROR'
        else:
            status = 'PENDING'
        return SingleRepoStatus(status, _GetErrorMessages(errors), rendering)
    elif source_time > rendering_time and source_time > sync_time:
        errors = _GetPathValue(obj, ['status', 'source', 'errors'], [])
        if errors:
            status = 'ERROR'
        else:
            status = 'PENDING'
        return SingleRepoStatus(status, _GetErrorMessages(errors), source)
    else:
        errors = _GetPathValue(obj, ['status', 'sync', 'errors'], [])
        if errors:
            status = 'ERROR'
        else:
            status = 'SYNCED'
        return SingleRepoStatus(status, _GetErrorMessages(errors), sync)