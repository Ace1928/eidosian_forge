import dataclasses
from typing import Collection
from werkzeug.datastructures import Headers
from werkzeug import http
from tensorboard.util import tb_logging
def _validate_csp_headers(self, headers):
    mime_type, _ = http.parse_options_header(headers.get('Content-Type'))
    if mime_type != _HTML_MIME_TYPE:
        return
    csp_texts = headers.get_all('Content-Security-Policy')
    policies = []
    for csp_text in csp_texts:
        policies += self._parse_serialized_csp(csp_text)
    self._validate_csp_policies(policies)