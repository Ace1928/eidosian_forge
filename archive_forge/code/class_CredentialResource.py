import hashlib
import flask
import http.client
from oslo_serialization import jsonutils
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone.credential import schema
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class CredentialResource(ks_flask.ResourceBase):
    collection_key = 'credentials'
    member_key = 'credential'

    @staticmethod
    def _blob_to_json(ref):
        blob = ref.get('blob')
        if isinstance(blob, dict):
            ref = ref.copy()
            ref['blob'] = jsonutils.dumps(blob)
        return ref

    def _validate_blob_json(self, ref):
        try:
            blob = jsonutils.loads(ref.get('blob'))
        except (ValueError, TabError):
            raise exception.ValidationError(message=_('Invalid blob in credential'))
        if not blob or not isinstance(blob, dict):
            raise exception.ValidationError(attribute='blob', target='credential')
        if blob.get('access') is None:
            raise exception.ValidationError(attribute='access', target='credential')
        return blob

    def _assign_unique_id(self, ref, trust_id=None, app_cred_id=None, access_token_id=None):
        if ref.get('type', '').lower() == 'ec2':
            blob = self._validate_blob_json(ref)
            ref = ref.copy()
            ref['id'] = hashlib.sha256(blob['access'].encode('utf8')).hexdigest()
            if trust_id is not None:
                blob['trust_id'] = trust_id
                ref['blob'] = jsonutils.dumps(blob)
            if app_cred_id is not None:
                blob['app_cred_id'] = app_cred_id
                ref['blob'] = jsonutils.dumps(blob)
            if access_token_id is not None:
                blob['access_token_id'] = access_token_id
                ref['blob'] = jsonutils.dumps(blob)
            return ref
        else:
            return super(CredentialResource, self)._assign_unique_id(ref)

    def _list_credentials(self):
        filters = ['user_id', 'type']
        if not self.oslo_context.system_scope:
            target = {'credential': {'user_id': self.oslo_context.user_id}}
        else:
            target = None
        ENFORCER.enforce_call(action='identity:list_credentials', filters=filters, target_attr=target)
        hints = self.build_driver_hints(filters)
        refs = PROVIDERS.credential_api.list_credentials(hints)
        filtered_refs = []
        for ref in refs:
            try:
                cred = PROVIDERS.credential_api.get_credential(ref['id'])
                ENFORCER.enforce_call(action='identity:get_credential', target_attr={'credential': cred})
                filtered_refs.append(ref)
            except exception.Forbidden:
                pass
        refs = filtered_refs
        refs = [self._blob_to_json(r) for r in refs]
        return self.wrap_collection(refs, hints=hints)

    def _get_credential(self, credential_id):
        ENFORCER.enforce_call(action='identity:get_credential', build_target=_build_target_enforcement)
        credential = PROVIDERS.credential_api.get_credential(credential_id)
        return self.wrap_member(self._blob_to_json(credential))

    def get(self, credential_id=None):
        if credential_id is None:
            return self._list_credentials()
        else:
            return self._get_credential(credential_id)

    def post(self):
        credential = self.request_body_json.get('credential', {})
        target = {}
        target['credential'] = credential
        ENFORCER.enforce_call(action='identity:create_credential', target_attr=target)
        validation.lazy_validate(schema.credential_create, credential)
        trust_id = getattr(self.oslo_context, 'trust_id', None)
        app_cred_id = getattr(self.auth_context['token'], 'application_credential_id', None)
        access_token_id = getattr(self.auth_context['token'], 'access_token_id', None)
        ref = self._assign_unique_id(self._normalize_dict(credential), trust_id=trust_id, app_cred_id=app_cred_id, access_token_id=access_token_id)
        ref = PROVIDERS.credential_api.create_credential(ref['id'], ref, initiator=self.audit_initiator)
        return (self.wrap_member(ref), http.client.CREATED)

    def _validate_blob_update_keys(self, credential, ref):
        if credential.get('type', '').lower() == 'ec2':
            new_blob = self._validate_blob_json(ref)
            old_blob = credential.get('blob')
            if isinstance(old_blob, str):
                old_blob = jsonutils.loads(old_blob)
            for key in ['trust_id', 'app_cred_id', 'access_token_id', 'access_id']:
                if old_blob.get(key) != new_blob.get(key):
                    message = _('%s can not be updated for credential') % key
                    raise exception.ValidationError(message=message)

    def patch(self, credential_id):
        ENFORCER.enforce_call(action='identity:update_credential', build_target=_build_target_enforcement)
        current = PROVIDERS.credential_api.get_credential(credential_id)
        credential = self.request_body_json.get('credential', {})
        validation.lazy_validate(schema.credential_update, credential)
        self._validate_blob_update_keys(current.copy(), credential.copy())
        self._require_matching_id(credential)
        target = {'credential': dict(current, **credential)}
        ENFORCER.enforce_call(action='identity:update_credential', target_attr=target)
        ref = PROVIDERS.credential_api.update_credential(credential_id, credential)
        return self.wrap_member(ref)

    def delete(self, credential_id):
        ENFORCER.enforce_call(action='identity:delete_credential', build_target=_build_target_enforcement)
        return (PROVIDERS.credential_api.delete_credential(credential_id, initiator=self.audit_initiator), http.client.NO_CONTENT)