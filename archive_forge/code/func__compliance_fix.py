import json
def _compliance_fix(response):
    token = json.loads(response.text)
    if token.get('token_type') in ['Application Access Token', 'User Access Token']:
        token['token_type'] = 'Bearer'
        fixed_token = json.dumps(token)
        response._content = fixed_token.encode()
    return response