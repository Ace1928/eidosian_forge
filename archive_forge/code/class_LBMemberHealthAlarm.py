from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine.resources.openstack.heat import none_resource
from heat.engine import support
from heat.engine import translation
class LBMemberHealthAlarm(AodhBaseActionsMixin, alarm_base.BaseAlarm):
    """A resource that implements a Loadbalancer Member Health Alarm.

    Allows setting alarms based on the health of load balancer pool members,
    where the health of a member is determined by the member reporting an
    operating_status of ERROR beyond an initial grace period after creation
    (120 seconds by default).
    """
    alarm_type = 'loadbalancer_member_health'
    support_status = support.SupportStatus(version='13.0.0')
    PROPERTIES = POOL, STACK, AUTOSCALING_GROUP_ID = ('pool', 'stack', 'autoscaling_group_id')
    RULE_PROPERTIES = POOL_ID, STACK_ID = ('pool_id', 'stack_id')
    properties_schema = {POOL: properties.Schema(properties.Schema.STRING, _('Name or ID of the loadbalancer pool for which the health of each member will be evaluated.'), update_allowed=True, required=True), STACK: properties.Schema(properties.Schema.STRING, _('Name or ID of the root / top level Heat stack containing the loadbalancer pool and members. An update will be triggered on the root Stack if an unhealthy member is detected in the loadbalancer pool.'), update_allowed=False, required=True), AUTOSCALING_GROUP_ID: properties.Schema(properties.Schema.STRING, _('ID of the Heat autoscaling group that contains the loadbalancer members. Unhealthy members will be marked as such before an update is triggered on the root stack.'), update_allowed=True, required=True)}
    properties_schema.update(alarm_base.common_properties_schema)

    def get_alarm_props(self, props):
        """Apply all relevant compatibility xforms."""
        kwargs = self.actions_to_urls(props)
        kwargs['type'] = self.alarm_type
        for prop in (self.POOL, self.STACK, self.AUTOSCALING_GROUP_ID):
            if prop in kwargs:
                del kwargs[prop]
        rule = {self.POOL_ID: props[self.POOL], self.STACK_ID: props[self.STACK], self.AUTOSCALING_GROUP_ID: props[self.AUTOSCALING_GROUP_ID]}
        kwargs['loadbalancer_member_health_rule'] = rule
        return kwargs

    def translation_rules(self, properties):
        translation_rules = [translation.TranslationRule(properties, translation.TranslationRule.RESOLVE, [self.POOL], client_plugin=self.client_plugin('octavia'), finder='get_pool')]
        return translation_rules