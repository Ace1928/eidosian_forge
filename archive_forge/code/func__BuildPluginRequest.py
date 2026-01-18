import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import baseauthenticator
def _BuildPluginRequest(self, app_id, challenge_data, origin):
    """Builds a JSON request in the form that the plugin expects."""
    client_data_map = {}
    encoded_challenges = []
    app_id_hash_encoded = self._Base64Encode(self._SHA256(app_id))
    for challenge_item in challenge_data:
        key = challenge_item['key']
        key_handle_encoded = self._Base64Encode(key.key_handle)
        raw_challenge = challenge_item['challenge']
        client_data_json = model.ClientData(model.ClientData.TYP_AUTHENTICATION, raw_challenge, origin).GetJson()
        challenge_hash_encoded = self._Base64Encode(self._SHA256(client_data_json))
        encoded_challenges.append({'appIdHash': app_id_hash_encoded, 'challengeHash': challenge_hash_encoded, 'keyHandle': key_handle_encoded, 'version': key.version})
        key_challenge_pair = (key_handle_encoded, challenge_hash_encoded)
        client_data_map[key_challenge_pair] = client_data_json
    signing_request = {'type': 'sign_helper_request', 'signData': encoded_challenges, 'timeoutSeconds': U2F_SIGNATURE_TIMEOUT_SECONDS, 'localAlways': True}
    return (client_data_map, json.dumps(signing_request))