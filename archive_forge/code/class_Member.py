import requests
from wandb_gql import gql
from wandb.apis.attrs import Attrs
class Member(Attrs):
    DELETE_MEMBER_MUTATION = gql('\n    mutation DeleteInvite($id: String, $entityName: String) {\n        deleteInvite(input: {id: $id, entityName: $entityName}) {\n            success\n        }\n    }\n  ')

    def __init__(self, client, team, attrs):
        super().__init__(attrs)
        self._client = client
        self.team = team

    def delete(self):
        """Remove a member from a team.

        Returns:
            Boolean indicating success
        """
        try:
            return self._client.execute(self.DELETE_MEMBER_MUTATION, {'id': self.id, 'entityName': self.team})['deleteInvite']['success']
        except requests.exceptions.HTTPError:
            return False

    def __repr__(self):
        return f'<Member {self.name} ({self.account_type})>'