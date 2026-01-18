import requests
from wandb_gql import gql
from wandb.apis.attrs import Attrs
class Team(Attrs):
    CREATE_TEAM_MUTATION = gql('\n    mutation CreateTeam($teamName: String!, $teamAdminUserName: String) {\n        createTeam(input: {teamName: $teamName, teamAdminUserName: $teamAdminUserName}) {\n            entity {\n                id\n                name\n                available\n                photoUrl\n                limits\n            }\n        }\n    }\n    ')
    CREATE_INVITE_MUTATION = gql('\n    mutation CreateInvite($entityName: String!, $email: String, $username: String, $admin: Boolean) {\n        createInvite(input: {entityName: $entityName, email: $email, username: $username, admin: $admin}) {\n            invite {\n                id\n                name\n                email\n                createdAt\n                toUser {\n                    name\n                }\n            }\n        }\n    }\n    ')
    TEAM_QUERY = gql('\n    query Entity($name: String!) {\n        entity(name: $name) {\n            id\n            name\n            available\n            photoUrl\n            readOnly\n            readOnlyAdmin\n            isTeam\n            privateOnly\n            storageBytes\n            codeSavingEnabled\n            defaultAccess\n            isPaid\n            members {\n                id\n                admin\n                pending\n                email\n                username\n                name\n                photoUrl\n                accountType\n                apiKey\n            }\n        }\n    }\n    ')
    CREATE_SERVICE_ACCOUNT_MUTATION = gql('\n    mutation CreateServiceAccount($entityName: String!, $description: String!) {\n        createServiceAccount(\n            input: {description: $description, entityName: $entityName}\n        ) {\n            user {\n                id\n            }\n        }\n    }\n    ')

    def __init__(self, client, name, attrs=None):
        super().__init__(attrs or {})
        self._client = client
        self.name = name
        self.load()

    @classmethod
    def create(cls, api, team, admin_username=None):
        """Create a new team.

        Arguments:
            api: (`Api`) The api instance to use
            team: (str) The name of the team
            admin_username: (str) optional username of the admin user of the team, defaults to the current user.

        Returns:
            A `Team` object
        """
        try:
            api.client.execute(cls.CREATE_TEAM_MUTATION, {'teamName': team, 'teamAdminUserName': admin_username})
        except requests.exceptions.HTTPError:
            pass
        return Team(api.client, team)

    def invite(self, username_or_email, admin=False):
        """Invite a user to a team.

        Arguments:
            username_or_email: (str) The username or email address of the user you want to invite
            admin: (bool) Whether to make this user a team admin, defaults to False

        Returns:
            True on success, False if user was already invited or didn't exist
        """
        variables = {'entityName': self.name, 'admin': admin}
        if '@' in username_or_email:
            variables['email'] = username_or_email
        else:
            variables['username'] = username_or_email
        try:
            self._client.execute(self.CREATE_INVITE_MUTATION, variables)
        except requests.exceptions.HTTPError:
            return False
        return True

    def create_service_account(self, description):
        """Create a service account for the team.

        Arguments:
            description: (str) A description for this service account

        Returns:
            The service account `Member` object, or None on failure
        """
        try:
            self._client.execute(self.CREATE_SERVICE_ACCOUNT_MUTATION, {'description': description, 'entityName': self.name})
            self.load(True)
            return self.members[-1]
        except requests.exceptions.HTTPError:
            return None

    def load(self, force=False):
        if force or not self._attrs:
            response = self._client.execute(self.TEAM_QUERY, {'name': self.name})
            self._attrs = response['entity']
            self._attrs['members'] = [Member(self._client, self.name, member) for member in self._attrs['members']]
        return self._attrs

    def __repr__(self):
        return f'<Team {self.name}>'