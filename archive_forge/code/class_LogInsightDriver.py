import json
import logging as log
from urllib import parse as urlparse
import netaddr
from oslo_concurrency.lockutils import synchronized
import requests
from osprofiler.drivers import base
from osprofiler import exc
class LogInsightDriver(base.Driver):
    """Driver for storing trace data in VMware vRealize Log Insight.

    The driver uses Log Insight ingest service to store trace data and uses
    the query service to retrieve it. The minimum required Log Insight version
    is 3.3.

    The connection string to initialize the driver should be of the format:
    loginsight://<username>:<password>@<loginsight-host>

    If the username or password contains the character ':' or '@', it must be
    escaped using URL encoding. For example, the connection string to connect
    to Log Insight server at 10.1.2.3 using username "osprofiler" and password
    "p@ssword" is: loginsight://osprofiler:p%40ssword@10.1.2.3
    """

    def __init__(self, connection_str, project=None, service=None, host=None, **kwargs):
        super(LogInsightDriver, self).__init__(connection_str, project=project, service=service, host=host)
        parsed_connection = urlparse.urlparse(connection_str)
        try:
            creds, host = parsed_connection.netloc.split('@')
            username, password = creds.split(':')
        except ValueError:
            raise ValueError("Connection string format is: loginsight://<username>:<password>@<loginsight-host>. If the username or password contains the character '@' or ':', it must be escaped using URL encoding.")
        username = urlparse.unquote(username)
        password = urlparse.unquote(password)
        self._client = LogInsightClient(host, username, password)
        self._client.login()

    @classmethod
    def get_name(cls):
        return 'loginsight'

    def notify(self, info):
        """Send trace to Log Insight server."""
        trace = info.copy()
        trace['project'] = self.project
        trace['service'] = self.service
        event = {'text': 'OSProfiler trace'}

        def _create_field(name, content):
            return {'name': name, 'content': content}
        event['fields'] = [_create_field('base_id', trace['base_id']), _create_field('trace_id', trace['trace_id']), _create_field('project', trace['project']), _create_field('service', trace['service']), _create_field('name', trace['name']), _create_field('trace', json.dumps(trace))]
        self._client.send_event(event)

    def get_report(self, base_id):
        """Retrieves and parses trace data from Log Insight.

        :param base_id: Trace base ID
        """
        response = self._client.query_events({'base_id': base_id})
        if 'events' in response:
            for event in response['events']:
                if 'fields' not in event:
                    continue
                for field in event['fields']:
                    if field['name'] == 'trace':
                        trace = json.loads(field['content'])
                        trace_id = trace['trace_id']
                        parent_id = trace['parent_id']
                        name = trace['name']
                        project = trace['project']
                        service = trace['service']
                        host = trace['info']['host']
                        timestamp = trace['timestamp']
                        self._append_results(trace_id, parent_id, name, project, service, host, timestamp, trace)
                        break
        return self._parse_results()