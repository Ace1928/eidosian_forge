import httplib2
Returns an http client that is authorized with the given credentials.

    Args:
        credentials (Union[
            google.auth.credentials.Credentials,
            oauth2client.client.Credentials]): The credentials to use.

    Returns:
        Union[httplib2.Http, google_auth_httplib2.AuthorizedHttp]: An
            authorized http client.
    