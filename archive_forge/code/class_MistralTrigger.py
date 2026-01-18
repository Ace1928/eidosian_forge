from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_serialization import jsonutils
class MistralTrigger(ZaqarSubscription):
    """A Zaqar subscription for triggering Mistral workflows.

    This Zaqar subscription type listens for messages in a queue and triggers a
    Mistral workflow execution each time one is received.

    The content of the Zaqar message is passed to the workflow in the
    environment with the name "notification", and thus is accessible from
    within the workflow as:

        <% env().notification %>

    Other environment variables can be set using the 'env' key in the params
    property.
    """
    support_status = support.SupportStatus(version='8.0.0', status=support.SUPPORTED)
    PROPERTIES = QUEUE_NAME, TTL, WORKFLOW_ID, PARAMS, INPUT = (ZaqarSubscription.QUEUE_NAME, ZaqarSubscription.TTL, 'workflow_id', 'params', 'input')
    properties_schema = {QUEUE_NAME: ZaqarSubscription.properties_schema[QUEUE_NAME], TTL: ZaqarSubscription.properties_schema[TTL], WORKFLOW_ID: properties.Schema(properties.Schema.STRING, _('UUID of the Mistral workflow to trigger.'), required=True, constraints=[constraints.CustomConstraint('mistral.workflow')], update_allowed=True), PARAMS: properties.Schema(properties.Schema.MAP, _('Parameters to pass to the Mistral workflow execution. The parameters depend on the workflow type.'), required=False, default={}, update_allowed=True), INPUT: properties.Schema(properties.Schema.MAP, _('Input values to pass to the Mistral workflow.'), required=False, default={}, update_allowed=True)}

    def _validate_subscriber(self):
        pass

    def _subscriber_url(self):
        mistral_client = self.client('mistral')
        manager = getattr(mistral_client.executions, 'client', mistral_client.executions)
        return 'trust+%s/executions' % manager.http_client.base_url

    def _subscription_options(self):
        params = dict(self.properties[self.PARAMS])
        params.setdefault('env', {})
        params['env']['notification'] = '$zaqar_message$'
        post_data = {self.WORKFLOW_ID: self.properties[self.WORKFLOW_ID], self.PARAMS: params, self.INPUT: self.properties[self.INPUT]}
        return {'post_data': jsonutils.dumps(post_data)}

    def parse_live_resource_data(self, resource_properties, resource_data):
        options = resource_data.get(self.OPTIONS, {})
        post_data = jsonutils.loads(options.get('post_data', '{}'))
        params = post_data.get(self.PARAMS, {})
        env = params.get('env', {})
        env.pop('notification', None)
        if not env:
            params.pop('env', None)
        return {self.QUEUE_NAME: resource_data.get(self.QUEUE_NAME), self.TTL: resource_data.get(self.TTL), self.WORKFLOW_ID: post_data.get(self.WORKFLOW_ID), self.PARAMS: params, self.INPUT: post_data.get(self.INPUT)}