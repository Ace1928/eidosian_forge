import datetime
import logging
from urllib import parse as urlparse
from osprofiler import _utils
def _append_results(self, trace_id, parent_id, name, project, service, host, timestamp, raw_payload=None):
    """Appends the notification to the dictionary of notifications.

        :param trace_id: UUID of current trace point
        :param parent_id: UUID of parent trace point
        :param name: name of operation
        :param project: project name
        :param service: service name
        :param host: host name or FQDN
        :param timestamp: Unicode-style timestamp matching the pattern
                          "%Y-%m-%dT%H:%M:%S.%f" , e.g. 2016-04-18T17:42:10.77
        :param raw_payload: raw notification without any filtering, with all
                            fields included
        """
    timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
    if trace_id not in self.result:
        self.result[trace_id] = {'info': {'name': name.split('-')[0], 'project': project, 'service': service, 'host': host}, 'trace_id': trace_id, 'parent_id': parent_id}
    self.result[trace_id]['info']['meta.raw_payload.%s' % name] = raw_payload
    if name.endswith('stop'):
        self.result[trace_id]['info']['finished'] = timestamp
        self.result[trace_id]['info']['exception'] = 'None'
        if raw_payload and 'info' in raw_payload:
            exc = raw_payload['info'].get('etype', 'None')
            self.result[trace_id]['info']['exception'] = exc
    else:
        self.result[trace_id]['info']['started'] = timestamp
        if not self.last_started_at or self.last_started_at < timestamp:
            self.last_started_at = timestamp
    if not self.started_at or self.started_at > timestamp:
        self.started_at = timestamp
    if not self.finished_at or self.finished_at < timestamp:
        self.finished_at = timestamp