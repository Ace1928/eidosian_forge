import json
def _non_zero_expiration(r):
    token = json.loads(r.text)
    if 'expires_in' in token and token['expires_in'] == 0:
        token['expires_in'] = 3600
    r._content = json.dumps(token).encode()
    return r