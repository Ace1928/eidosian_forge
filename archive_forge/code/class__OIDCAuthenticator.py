from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Optional
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
@dataclass
class _OIDCAuthenticator:
    username: str
    properties: _OIDCProperties
    refresh_token: Optional[str] = field(default=None)
    access_token: Optional[str] = field(default=None)
    idp_info: Optional[dict] = field(default=None)
    token_gen_id: int = field(default=0)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_current_token(self, use_callback: bool=True) -> Optional[str]:
        properties = self.properties
        cb = properties.request_token_callback if use_callback else None
        cb_type = 'human'
        prev_token = self.access_token
        if prev_token:
            return prev_token
        if not use_callback and (not prev_token):
            return None
        if not prev_token and cb is not None:
            with self.lock:
                new_token = self.access_token
                if new_token != prev_token:
                    return new_token
                if cb_type == 'human':
                    context = {'timeout_seconds': CALLBACK_TIMEOUT_SECONDS, 'version': CALLBACK_VERSION, 'refresh_token': self.refresh_token}
                    resp = cb(self.idp_info, context)
                    self.validate_request_token_response(resp)
                self.token_gen_id += 1
        return self.access_token

    def validate_request_token_response(self, resp: Mapping[str, Any]) -> None:
        if not isinstance(resp, dict):
            raise ValueError('OIDC callback returned invalid result')
        if 'access_token' not in resp:
            raise ValueError('OIDC callback did not return an access_token')
        expected = ['access_token', 'refresh_token', 'expires_in_seconds']
        for key in resp:
            if key not in expected:
                raise ValueError(f'Unexpected field in callback result "{key}"')
        self.access_token = resp['access_token']
        self.refresh_token = resp.get('refresh_token')

    def principal_step_cmd(self) -> SON[str, Any]:
        """Get a SASL start command with an optional principal name"""
        payload = {}
        principal_name = self.username
        if principal_name:
            payload['n'] = principal_name
        return SON([('saslStart', 1), ('mechanism', 'MONGODB-OIDC'), ('payload', Binary(bson.encode(payload))), ('autoAuthorize', 1)])

    def auth_start_cmd(self, use_callback: bool=True) -> Optional[SON[str, Any]]:
        if self.idp_info is None:
            return self.principal_step_cmd()
        token = self.get_current_token(use_callback)
        if not token:
            return None
        bin_payload = Binary(bson.encode({'jwt': token}))
        return SON([('saslStart', 1), ('mechanism', 'MONGODB-OIDC'), ('payload', bin_payload)])

    def run_command(self, conn: Connection, cmd: MutableMapping[str, Any]) -> Optional[Mapping[str, Any]]:
        try:
            return conn.command('$external', cmd, no_reauth=True)
        except OperationFailure:
            self.access_token = None
            raise

    def reauthenticate(self, conn: Connection) -> Optional[Mapping[str, Any]]:
        """Handle a reauthenticate from the server."""
        prev_id = conn.oidc_token_gen_id or 0
        if prev_id < self.token_gen_id and self.access_token:
            try:
                return self.authenticate(conn)
            except OperationFailure:
                pass
        self.access_token = None
        prev_idp_info = self.idp_info
        self.idp_info = None
        cmd = self.principal_step_cmd()
        resp = self.run_command(conn, cmd)
        assert resp is not None
        server_resp: dict = bson.decode(resp['payload'])
        if 'issuer' in server_resp:
            self.idp_info = server_resp
        if self.idp_info != prev_idp_info:
            self.access_token = None
            self.refresh_token = None
        if self.refresh_token:
            try:
                return self.finish_auth(resp, conn)
            except OperationFailure:
                self.refresh_token = None
                return self.authenticate(conn)
        return self.finish_auth(resp, conn)

    def authenticate(self, conn: Connection) -> Optional[Mapping[str, Any]]:
        ctx = conn.auth_ctx
        cmd = None
        if ctx and ctx.speculate_succeeded():
            resp = ctx.speculative_authenticate
        else:
            cmd = self.auth_start_cmd()
            assert cmd is not None
            resp = self.run_command(conn, cmd)
        assert resp is not None
        if resp['done']:
            conn.oidc_token_gen_id = self.token_gen_id
            return None
        server_resp: dict = bson.decode(resp['payload'])
        if 'issuer' in server_resp:
            self.idp_info = server_resp
        return self.finish_auth(resp, conn)

    def finish_auth(self, orig_resp: Mapping[str, Any], conn: Connection) -> Optional[Mapping[str, Any]]:
        conversation_id = orig_resp['conversationId']
        token = self.get_current_token()
        conn.oidc_token_gen_id = self.token_gen_id
        bin_payload = Binary(bson.encode({'jwt': token}))
        cmd = SON([('saslContinue', 1), ('conversationId', conversation_id), ('payload', bin_payload)])
        resp = self.run_command(conn, cmd)
        assert resp is not None
        if not resp['done']:
            raise OperationFailure('SASL conversation failed to complete.')
        return resp