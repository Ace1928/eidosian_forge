from json import loads, dumps
def _missing_error(r):
    token = loads(r.text)
    if 'errors' in token:
        token['error'] = token['errors'][0]['errorType']
    r._content = dumps(token).encode()
    return r