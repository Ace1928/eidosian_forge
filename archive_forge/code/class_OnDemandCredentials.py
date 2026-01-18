import copy
import datetime
import json
import cachetools
import six
from six.moves import urllib
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import crypt
from google.auth import exceptions
import google.auth.credentials
class OnDemandCredentials(google.auth.credentials.Signing, google.auth.credentials.CredentialsWithQuotaProject):
    """On-demand JWT credentials.

    Like :class:`Credentials`, this class uses a JWT as the bearer token for
    authentication. However, this class does not require the audience at
    construction time. Instead, it will generate a new token on-demand for
    each request using the request URI as the audience. It caches tokens
    so that multiple requests to the same URI do not incur the overhead
    of generating a new token every time.

    This behavior is especially useful for `gRPC`_ clients. A gRPC service may
    have multiple audience and gRPC clients may not know all of the audiences
    required for accessing a particular service. With these credentials,
    no knowledge of the audiences is required ahead of time.

    .. _grpc: http://www.grpc.io/
    """

    def __init__(self, signer, issuer, subject, additional_claims=None, token_lifetime=_DEFAULT_TOKEN_LIFETIME_SECS, max_cache_size=_DEFAULT_MAX_CACHE_SIZE, quota_project_id=None):
        """
        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            issuer (str): The `iss` claim.
            subject (str): The `sub` claim.
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT payload.
            token_lifetime (int): The amount of time in seconds for
                which the token is valid. Defaults to 1 hour.
            max_cache_size (int): The maximum number of JWT tokens to keep in
                cache. Tokens are cached using :class:`cachetools.LRUCache`.
            quota_project_id (Optional[str]): The project ID used for quota
                and billing.

        """
        super(OnDemandCredentials, self).__init__()
        self._signer = signer
        self._issuer = issuer
        self._subject = subject
        self._token_lifetime = token_lifetime
        self._quota_project_id = quota_project_id
        if additional_claims is None:
            additional_claims = {}
        self._additional_claims = additional_claims
        self._cache = cachetools.LRUCache(maxsize=max_cache_size)

    @classmethod
    def _from_signer_and_info(cls, signer, info, **kwargs):
        """Creates an OnDemandCredentials instance from a signer and service
        account info.

        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            info (Mapping[str, str]): The service account info.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.OnDemandCredentials: The constructed credentials.

        Raises:
            google.auth.exceptions.MalformedError: If the info is not in the expected format.
        """
        kwargs.setdefault('subject', info['client_email'])
        kwargs.setdefault('issuer', info['client_email'])
        return cls(signer, **kwargs)

    @classmethod
    def from_service_account_info(cls, info, **kwargs):
        """Creates an OnDemandCredentials instance from a dictionary.

        Args:
            info (Mapping[str, str]): The service account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.OnDemandCredentials: The constructed credentials.

        Raises:
            google.auth.exceptions.MalformedError: If the info is not in the expected format.
        """
        signer = _service_account_info.from_dict(info, require=['client_email'])
        return cls._from_signer_and_info(signer, info, **kwargs)

    @classmethod
    def from_service_account_file(cls, filename, **kwargs):
        """Creates an OnDemandCredentials instance from a service account .json
        file in Google format.

        Args:
            filename (str): The path to the service account .json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.OnDemandCredentials: The constructed credentials.
        """
        info, signer = _service_account_info.from_filename(filename, require=['client_email'])
        return cls._from_signer_and_info(signer, info, **kwargs)

    @classmethod
    def from_signing_credentials(cls, credentials, **kwargs):
        """Creates a new :class:`google.auth.jwt.OnDemandCredentials` instance
        from an existing :class:`google.auth.credentials.Signing` instance.

        The new instance will use the same signer as the existing instance and
        will use the existing instance's signer email as the issuer and
        subject by default.

        Example::

            svc_creds = service_account.Credentials.from_service_account_file(
                'service_account.json')
            jwt_creds = jwt.OnDemandCredentials.from_signing_credentials(
                svc_creds)

        Args:
            credentials (google.auth.credentials.Signing): The credentials to
                use to construct the new credentials.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.Credentials: A new Credentials instance.
        """
        kwargs.setdefault('issuer', credentials.signer_email)
        kwargs.setdefault('subject', credentials.signer_email)
        return cls(credentials.signer, **kwargs)

    def with_claims(self, issuer=None, subject=None, additional_claims=None):
        """Returns a copy of these credentials with modified claims.

        Args:
            issuer (str): The `iss` claim. If unspecified the current issuer
                claim will be used.
            subject (str): The `sub` claim. If unspecified the current subject
                claim will be used.
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT payload. This will be merged with the current
                additional claims.

        Returns:
            google.auth.jwt.OnDemandCredentials: A new credentials instance.
        """
        new_additional_claims = copy.deepcopy(self._additional_claims)
        new_additional_claims.update(additional_claims or {})
        return self.__class__(self._signer, issuer=issuer if issuer is not None else self._issuer, subject=subject if subject is not None else self._subject, additional_claims=new_additional_claims, max_cache_size=self._cache.maxsize, quota_project_id=self._quota_project_id)

    @_helpers.copy_docstring(google.auth.credentials.CredentialsWithQuotaProject)
    def with_quota_project(self, quota_project_id):
        return self.__class__(self._signer, issuer=self._issuer, subject=self._subject, additional_claims=self._additional_claims, max_cache_size=self._cache.maxsize, quota_project_id=quota_project_id)

    @property
    def valid(self):
        """Checks the validity of the credentials.

        These credentials are always valid because it generates tokens on
        demand.
        """
        return True

    def _make_jwt_for_audience(self, audience):
        """Make a new JWT for the given audience.

        Args:
            audience (str): The intended audience.

        Returns:
            Tuple[bytes, datetime]: The encoded JWT and the expiration.
        """
        now = _helpers.utcnow()
        lifetime = datetime.timedelta(seconds=self._token_lifetime)
        expiry = now + lifetime
        payload = {'iss': self._issuer, 'sub': self._subject, 'iat': _helpers.datetime_to_secs(now), 'exp': _helpers.datetime_to_secs(expiry), 'aud': audience}
        payload.update(self._additional_claims)
        jwt = encode(self._signer, payload)
        return (jwt, expiry)

    def _get_jwt_for_audience(self, audience):
        """Get a JWT For a given audience.

        If there is already an existing, non-expired token in the cache for
        the audience, that token is used. Otherwise, a new token will be
        created.

        Args:
            audience (str): The intended audience.

        Returns:
            bytes: The encoded JWT.
        """
        token, expiry = self._cache.get(audience, (None, None))
        if token is None or expiry < _helpers.utcnow():
            token, expiry = self._make_jwt_for_audience(audience)
            self._cache[audience] = (token, expiry)
        return token

    def refresh(self, request):
        """Raises an exception, these credentials can not be directly
        refreshed.

        Args:
            request (Any): Unused.

        Raises:
            google.auth.RefreshError
        """
        raise exceptions.RefreshError('OnDemandCredentials can not be directly refreshed.')

    def before_request(self, request, method, url, headers):
        """Performs credential-specific before request logic.

        Args:
            request (Any): Unused. JWT credentials do not need to make an
                HTTP request to refresh.
            method (str): The request's HTTP method.
            url (str): The request's URI. This is used as the audience claim
                when generating the JWT.
            headers (Mapping): The request's headers.
        """
        parts = urllib.parse.urlsplit(url)
        audience = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
        token = self._get_jwt_for_audience(audience)
        self.apply(headers, token=token)

    @_helpers.copy_docstring(google.auth.credentials.Signing)
    def sign_bytes(self, message):
        return self._signer.sign(message)

    @property
    @_helpers.copy_docstring(google.auth.credentials.Signing)
    def signer_email(self):
        return self._issuer

    @property
    @_helpers.copy_docstring(google.auth.credentials.Signing)
    def signer(self):
        return self._signer