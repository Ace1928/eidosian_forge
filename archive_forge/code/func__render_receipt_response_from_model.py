import flask
import http.client
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import provider_api
from keystone import exception
def _render_receipt_response_from_model(receipt):
    receipt_reference = {'receipt': {'methods': receipt.methods, 'user': {'id': receipt.user['id'], 'name': receipt.user['name'], 'domain': {'id': receipt.user_domain['id'], 'name': receipt.user_domain['name']}}, 'expires_at': receipt.expires_at, 'issued_at': receipt.issued_at}, 'required_auth_methods': receipt.required_methods}
    return receipt_reference