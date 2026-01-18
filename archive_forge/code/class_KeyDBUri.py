import os
import pathlib
import typing
import threading
import functools
import hashlib
from aiokeydb.types.compat import validator, root_validator, Field
from aiokeydb.types.compat import BaseSettings as _BaseSettings
from aiokeydb.types.compat import BaseModel as _BaseModel
from pydantic.networks import AnyUrl
class KeyDBUri(BaseModel):
    dsn: KeyDBDsn

    @lazyproperty
    def host(self):
        return self.dsn.host

    @lazyproperty
    def port(self):
        return self.dsn.port

    @lazyproperty
    def path(self):
        return self.dsn.path

    @lazyproperty
    def username(self):
        return self.dsn.user

    @lazyproperty
    def password(self):
        return self.dsn.password

    @lazyproperty
    def db_id(self):
        return int(self.dsn.path[1:]) if self.dsn.path else None

    @lazyproperty
    def ssl(self):
        return self.dsn.scheme in {'rediss', 'keydbs'}

    @lazyproperty
    def uri(self):
        return str(self.dsn)

    @lazyproperty
    def connection(self):
        return str(self.dsn)

    @lazyproperty
    def uri_no_auth(self):
        if self.has_auth:
            return str(self.dsn).replace(f'{self.auth_str}', '***')
        return str(self.dsn)

    @lazyproperty
    def auth_str(self):
        if self.dsn.user:
            return f'{self.dsn.user}:{self.password}' if self.password else f'{self.dsn.user}'
        return f':{self.dsn.password}' if self.dsn.password else ''

    @lazyproperty
    def has_auth(self):
        return self.dsn.user or self.dsn.password

    def __str__(self):
        return f'{self.uri_no_auth}'

    def __repr__(self):
        return f'<KeyDBUri {self.uri_no_auth}>'

    @lazyproperty
    def key(self):
        """
        Returns the hashkey for the uri
        """
        return hashlib.md5(self.uri.encode('ascii')).hexdigest()

    @lazyproperty
    def connection_args(self) -> typing.List[str]:
        """
        Returns the connection arguments for CLI usage
        """
        args = []
        if self.host:
            args.append(f'-h {self.host}')
        if self.port:
            args.append(f'-p {self.port}')
        if self.username:
            args.append(f'--user {self.username}')
        if self.password:
            args.append(f'-a {self.password} --no-auth-warning')
        return args