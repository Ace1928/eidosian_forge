import json
import six
def _validate_clientsecrets(clientsecrets_dict):
    """Validate parsed client secrets from a file.

    Args:
        clientsecrets_dict: dict, a dictionary holding the client secrets.

    Returns:
        tuple, a string of the client type and the information parsed
        from the file.
    """
    _INVALID_FILE_FORMAT_MSG = 'Invalid file format. See https://developers.google.com/api-client-library/python/guide/aaa_client_secrets'
    if clientsecrets_dict is None:
        raise InvalidClientSecretsError(_INVALID_FILE_FORMAT_MSG)
    try:
        (client_type, client_info), = clientsecrets_dict.items()
    except (ValueError, AttributeError):
        raise InvalidClientSecretsError(_INVALID_FILE_FORMAT_MSG + ' Expected a JSON object with a single property for a "web" or "installed" application')
    if client_type not in VALID_CLIENT:
        raise InvalidClientSecretsError('Unknown client type: {0}.'.format(client_type))
    for prop_name in VALID_CLIENT[client_type]['required']:
        if prop_name not in client_info:
            raise InvalidClientSecretsError('Missing property "{0}" in a client type of "{1}".'.format(prop_name, client_type))
    for prop_name in VALID_CLIENT[client_type]['string']:
        if client_info[prop_name].startswith('[['):
            raise InvalidClientSecretsError('Property "{0}" is not configured.'.format(prop_name))
    return (client_type, client_info)