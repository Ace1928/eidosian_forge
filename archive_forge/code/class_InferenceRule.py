import fixtures
import uuid
class InferenceRule(Base):

    def __init__(self, client, prior_role, implied_role):
        super(InferenceRule, self).__init__(client)
        self.prior_role = prior_role
        self.implied_role = implied_role

    def setUp(self):
        super(InferenceRule, self).setUp()
        self.ref = {'prior_role': self.prior_role, 'implied_role': self.implied_role}
        self.entity = self.client.inference_rules.create(**self.ref)
        self.addCleanup(self.client.inference_rules.delete, self.prior_role, self.implied_role)