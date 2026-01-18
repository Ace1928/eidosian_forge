from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def _authenticate_aws(credentials: MongoCredential, conn: Connection) -> None:
    """Authenticate using MONGODB-AWS."""
    if not _HAVE_MONGODB_AWS:
        raise ConfigurationError("MONGODB-AWS authentication requires pymongo-auth-aws: install with: python -m pip install 'pymongo[aws]'")
    if conn.max_wire_version < 9:
        raise ConfigurationError('MONGODB-AWS authentication requires MongoDB version 4.4 or later')
    try:
        ctx = _AwsSaslContext(AwsCredential(credentials.username, credentials.password, credentials.mechanism_properties.aws_session_token))
        client_payload = ctx.step(None)
        client_first = SON([('saslStart', 1), ('mechanism', 'MONGODB-AWS'), ('payload', client_payload)])
        server_first = conn.command('$external', client_first)
        res = server_first
        for _ in range(10):
            client_payload = ctx.step(res['payload'])
            cmd = SON([('saslContinue', 1), ('conversationId', server_first['conversationId']), ('payload', client_payload)])
            res = conn.command('$external', cmd)
            if res['done']:
                break
    except PyMongoAuthAwsError as exc:
        set_cached_credentials(None)
        raise OperationFailure(f'{exc} (pymongo-auth-aws version {pymongo_auth_aws.__version__})') from None
    except Exception:
        set_cached_credentials(None)
        raise