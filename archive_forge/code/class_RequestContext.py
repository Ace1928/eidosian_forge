from oslo_context import context as oslo_context
class RequestContext(oslo_context.RequestContext):

    def __init__(self, **kwargs):
        self.username = kwargs.pop('username', None)
        self.project_tag_name = kwargs.pop('project_tag_name', None)
        self.is_delegated_auth = kwargs.pop('is_delegated_auth', False)
        self.trust_id = kwargs.pop('trust_id', None)
        self.trustor_id = kwargs.pop('trustor_id', None)
        self.trustee_id = kwargs.pop('trustee_id', None)
        self.oauth_consumer_id = kwargs.pop('oauth_consumer_id', None)
        self.oauth_access_token_id = kwargs.pop('oauth_access_token_id', None)
        self.authenticated = kwargs.pop('authenticated', False)
        super(RequestContext, self).__init__(**kwargs)

    def to_policy_values(self):
        """Add keystone-specific policy values to policy representation.

        This method converts generic policy values to a dictionary form using
        the base implementation from oslo_context.context.RequestContext.
        Afterwards, it is going to pull keystone-specific values off the
        context and represent them as items in the policy values dictionary.
        This is because keystone uses default policies that rely on these
        values, so we need to guarantee they are present during policy
        enforcement if they are present on the context object.

        This method is automatically called in
        oslo_policy.policy.Enforcer.enforce() if oslo.policy knows it's dealing
        with a context object.

        """
        values = super(RequestContext, self).to_policy_values()
        values['token'] = self.token_reference['token']
        values['domain_id'] = self.domain_id if self.domain_id else None
        return values