from keystoneauth1.identity.v3 import base
A plugin for authenticating with an application credential.

    :param string auth_url: Identity service endpoint for authentication.
    :param string application_credential_secret: Application credential secret.
    :param string application_credential_id: Application credential ID.
    :param string application_credential_name: Application credential name.
    :param string username: Username for authentication.
    :param string user_id: User ID for authentication.
    :param string user_domain_id: User's domain ID for authentication.
    :param string user_domain_name: User's domain name for authentication.
    :param bool reauthenticate: Allow fetching a new token if the current one
                                is going to expire. (optional) default True
    