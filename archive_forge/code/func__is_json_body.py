import json
import os
import betamax.serializers.base
import yaml
def _is_json_body(interaction):
    content_type = interaction['headers'].get('Content-Type', [])
    return 'application/json' in content_type