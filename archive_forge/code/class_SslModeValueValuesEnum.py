from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SslModeValueValuesEnum(_messages.Enum):
    """Specify how SSL/TLS is enforced in database connections. If you must
    use the `require_ssl` flag for backward compatibility, then only the
    following value pairs are valid: For PostgreSQL and MySQL: *
    `ssl_mode=ALLOW_UNENCRYPTED_AND_ENCRYPTED` and `require_ssl=false` *
    `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=false` *
    `ssl_mode=TRUSTED_CLIENT_CERTIFICATE_REQUIRED` and `require_ssl=true` For
    SQL Server: * `ssl_mode=ALLOW_UNENCRYPTED_AND_ENCRYPTED` and
    `require_ssl=false` * `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=true` The
    value of `ssl_mode` gets priority over the value of `require_ssl`. For
    example, for the pair `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=false`,
    the `ssl_mode=ENCRYPTED_ONLY` means only accept SSL connections, while the
    `require_ssl=false` means accept both non-SSL and SSL connections. MySQL
    and PostgreSQL databases respect `ssl_mode` in this case and accept only
    SSL connections.

    Values:
      SSL_MODE_UNSPECIFIED: The SSL mode is unknown.
      ALLOW_UNENCRYPTED_AND_ENCRYPTED: Allow non-SSL/non-TLS and SSL/TLS
        connections. For SSL/TLS connections, the client certificate won't be
        verified. When this value is used, the legacy `require_ssl` flag must
        be false or cleared to avoid the conflict between values of two flags.
      ENCRYPTED_ONLY: Only allow connections encrypted with SSL/TLS. When this
        value is used, the legacy `require_ssl` flag must be false or cleared
        to avoid the conflict between values of two flags.
      TRUSTED_CLIENT_CERTIFICATE_REQUIRED: Only allow connections encrypted
        with SSL/TLS and with valid client certificates. When this value is
        used, the legacy `require_ssl` flag must be true or cleared to avoid
        the conflict between values of two flags. PostgreSQL clients or users
        that connect using IAM database authentication must use either the
        [Cloud SQL Auth
        Proxy](https://cloud.google.com/sql/docs/postgres/connect-auth-proxy)
        or [Cloud SQL
        Connectors](https://cloud.google.com/sql/docs/postgres/connect-
        connectors) to enforce client identity verification. This value is not
        applicable to SQL Server.
    """
    SSL_MODE_UNSPECIFIED = 0
    ALLOW_UNENCRYPTED_AND_ENCRYPTED = 1
    ENCRYPTED_ONLY = 2
    TRUSTED_CLIENT_CERTIFICATE_REQUIRED = 3