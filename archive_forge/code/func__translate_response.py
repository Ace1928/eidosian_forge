import uuid
from openstack import resource
def _translate_response(self, response, has_body=True):
    super(Claim, self)._translate_response(response, has_body=has_body)
    if has_body and self.location:
        self.id = self.location.split('claims/')[1]