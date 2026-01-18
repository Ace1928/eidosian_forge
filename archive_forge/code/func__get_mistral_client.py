import json
import uuid
from mistralclient.api.client import client as mistral_client
from troveclient import base
from troveclient import common
def _get_mistral_client(self):
    if hasattr(self.api.client, 'auth'):
        auth_url = self.api.client.auth.auth_url
        user = self.api.client.auth._username
        key = self.api.client.auth._password
        tenant_name = self.api.client.auth._project_name
    else:
        auth_url = self.api.client.auth_url
        user = self.api.client.user
        key = self.api.client.password
        tenant_name = self.api.client.projectid
    return mistral_client(auth_url=auth_url, username=user, api_key=key, project_name=tenant_name)