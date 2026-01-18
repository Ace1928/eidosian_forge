from heat.common import exception
from heat.engine import constraints
class KeystoneBaseConstraint(constraints.BaseCustomConstraint):
    resource_client_name = CLIENT_NAME
    entity = None

    def validate_with_client(self, client, resource_id):
        if resource_id == '':
            raise exception.EntityNotFound(entity=self.entity, name=resource_id)
        super(KeystoneBaseConstraint, self).validate_with_client(client, resource_id)