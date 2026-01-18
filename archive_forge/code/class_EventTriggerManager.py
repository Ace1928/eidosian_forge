from oslo_serialization import jsonutils
from mistralclient.api import base
class EventTriggerManager(base.ResourceManager):
    resource_class = EventTrigger

    def create(self, name, workflow_id, exchange, topic, event, workflow_input=None, workflow_params=None):
        self._ensure_not_empty(name=name, workflow_id=workflow_id)
        data = {'workflow_id': workflow_id, 'name': name, 'exchange': exchange, 'topic': topic, 'event': event}
        if workflow_input:
            data.update({'workflow_input': jsonutils.dumps(workflow_input)})
        if workflow_params:
            data.update({'workflow_params': jsonutils.dumps(workflow_params)})
        return self._create('/event_triggers', data)

    def list(self, marker='', limit=None, sort_keys='', sort_dirs='', fields='', **filters):
        query_string = self._build_query_params(marker=marker, limit=limit, sort_keys=sort_keys, sort_dirs=sort_dirs, fields=fields, filters=filters)
        return self._list('/event_triggers%s' % query_string, response_key='event_triggers')

    def get(self, id):
        self._ensure_not_empty(id=id)
        return self._get('/event_triggers/%s' % id)

    def delete(self, id):
        self._ensure_not_empty(id=id)
        self._delete('/event_triggers/%s' % id)