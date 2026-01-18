import os
import time
import asyncio
import contextlib
from lazyops.imports._sqlalchemy import require_sql
from sqlalchemy import create_engine, event, exc
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import Session, scoped_session
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine, async_scoped_session
from lazyops.utils.logs import logger
from lazyops.utils import Json
from lazyops.types import BaseModel, lazyproperty, BaseSettings, Field
from typing import Any, Generator, AsyncGenerator, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from pydantic.networks import PostgresDsn
from lazyops.libs.psqldb.retry import reconnecting_engine
from lazyops.utils.helpers import import_string
class PostgresDBMeta(type):
    _settings: Optional[SettingsT] = None
    _ctx: Optional[Context] = None
    _uri: Optional[str] = None
    _config: Optional[Dict[str, Any]] = None
    scheme: Optional[str] = 'postgresql+psycopg2'
    async_scheme: Optional[str] = 'postgresql+asyncpg'
    get_settings_callable: Optional[Callable] = None

    @property
    def settings(cls) -> SettingsT:
        """
        The settings for the database
        """
        if cls._settings is None:
            cls._settings = cls.get_settings()
            if cls._settings:
                if getattr(cls._settings, 'postgres_scheme', None):
                    cls.scheme = cls._settings.postgres_scheme
                if getattr(cls._settings, 'postgres_async_scheme', None):
                    cls.async_scheme = cls._settings.postgres_async_scheme
        return cls._settings

    def get_settings(cls, settings: Optional[SettingsT]=None) -> SettingsT:
        """
        Helper method to override and get/set the settings
        """
        if settings is not None:
            cls.set_settings(settings)
        elif cls.get_settings_callable is not None:
            cls.set_settings(cls.get_settings_callable())
        return cls._settings

    def set_settings(cls, settings: SettingsT):
        """
        Sets the settings
        """
        cls._settings = settings

    @property
    def uri(cls) -> Union[str, SafePostgresDsn]:
        """
        Returns the uri
        """
        if not cls._uri and cls.settings:
            if getattr(cls.settings, 'pg_uri', None):
                cls._uri = cls.settings.pg_uri
            elif getattr(cls.settings, 'postgres_url', None):
                cls._uri = uri_builder(cls.settings.postgres_url, scheme=cls.scheme)
            elif (postgres_host := getattr(cls.settings, 'postgres_host', os.getenv('POSTGRES_HOST'))):
                base_uri = f'{postgres_host}:{getattr(cls.settings, 'postgres_port', 5432)}/{getattr(cls.settings, 'postgres_db', 'postgres')}'
                if (postgres_user := getattr(cls.settings, 'postgres_user', os.getenv('POSTGRES_USER'))):
                    base_uri = f'{postgres_user}:{cls.settings.postgres_password}@{base_uri}' if getattr(cls.settings, 'postgres_password', None) else f'{postgres_user}@{base_uri}'
                cls._uri = uri_builder(base_uri, scheme=cls.scheme)
        if not cls._uri:
            cls._uri = uri_builder(os.getenv('POSTGRES_URI', 'postgres@127.0.0.1:5432/postgres'), scheme=cls.scheme)
        return cls._uri

    @property
    def safe_uri(cls) -> str:
        """
        Returns the safe uri for logging
        """
        return cls.uri.safestr

    @property
    def pg_admin_user(cls) -> str:
        """
        Returns the admin user
        """
        if (admin_user := cls.config.get('pg_admin_user', os.getenv('POSTGRES_ADMIN_USER', os.getenv('POSTGRES_USER')))):
            return admin_user
        return cls.uri.user

    @property
    def pg_admin_password(cls) -> str:
        """
        Returns the admin password
        """
        if (admin_password := cls.config.get('pg_admin_password', os.getenv('POSTGRES_ADMIN_PASSWORD', os.getenv('POSTGRES_PASSWORD')))):
            return admin_password
        return cls.uri.password

    @property
    def pg_admin_db(cls) -> str:
        """
        Returns the admin db
        """
        if (admin_db := cls.config.get('pg_admin_db', os.getenv('POSTGRES_ADMIN_DB', os.getenv('POSTGRES_DB')))):
            return admin_db
        return cls.uri.path[1:]

    @property
    def admin_uri(cls) -> SafePostgresDsn:
        """
        Returns the admin uri
        """
        uri = f'{cls.uri.host}:{cls.uri.port}/{cls.pg_admin_db}'
        auth = f'{cls.pg_admin_user}'
        if cls.pg_admin_password:
            auth += f':{cls.pg_admin_password}'
        uri = f'{auth}@{uri}'
        return uri_builder(uri, scheme=cls.scheme)

    def get_admin_uri(cls, host: Optional[str]=None, port: Optional[int]=None, user: Optional[str]=None, password: Optional[str]=None, db: Optional[str]=None) -> SafePostgresDsn:
        """
        Returns the admin uri
        """
        uri = f'{host or cls.uri.host}:{port or cls.uri.port}/{db or cls.pg_admin_db}'
        auth = f'{user or cls.pg_admin_user}'
        if password or cls.pg_admin_password:
            auth += f':{password or cls.pg_admin_password}'
        uri = f'{auth}@{uri}'
        return uri_builder(uri, scheme=cls.scheme)

    @property
    def config(cls) -> Dict[str, Any]:
        """
        Returns the config
        """
        if not cls._config:
            cls._config = {}
            if cls.settings:
                if getattr(cls.settings, 'pg_config', None):
                    cls._config = cls.settings.pg_config
                elif getattr(cls.settings, 'postgres_config', None):
                    cls._config = cls.settings.postgres_config
        return cls._config

    @property
    def ctx(cls) -> Context:
        """
        Returns the context
        """
        if not cls._ctx:
            cls._ctx = Context.from_uri(cls.uri, settings=cls.settings, config=cls.config, scheme=cls.scheme, async_scheme=cls.async_scheme)
        return cls._ctx

    @property
    def engine(cls) -> Engine:
        """
        Returns the read-write engine
        """
        return cls.ctx.engine

    @property
    def engine_ro(cls) -> Optional[Engine]:
        """
        Returns the read-only engine
        """
        return cls.ctx.engine_ro

    @property
    def async_engine(cls) -> AsyncEngine:
        """
        Returns the read-write async engine
        """
        return cls.ctx.async_engine

    @property
    def async_engine_ro(cls) -> Optional[AsyncEngine]:
        """
        Returns the read-only async engine
        """
        return cls.ctx.async_engine_ro

    def get_sess(cls, ro: Optional[bool]=False, future: bool=True, **kwargs) -> Session:
        """
        Returns a session
        """
        return cls.ctx.get_sess(ro=ro, future=future, **kwargs)

    def get_async_sess(cls, ro: Optional[bool]=False, **kwargs) -> AsyncSession:
        """
        Returns an async session
        """
        return cls.ctx.get_async_sess(ro=ro, **kwargs)

    @contextlib.contextmanager
    def session(cls, ro: Optional[bool]=False, future: bool=True, session: Optional[Session]=None, **kwargs) -> Generator[Session, None, None]:
        """
        Context manager for database session
        """
        with cls.ctx.get_session(ro=ro, future=future, session=session, **kwargs) as sess:
            yield sess

    @contextlib.asynccontextmanager
    async def async_session(cls, ro: Optional[bool]=False, retries: Optional[int]=None, retry_interval: Optional[float]=None, session: Optional[AsyncSession]=None, **kwargs) -> Generator[AsyncSession, None, None]:
        """
        Async context manager for database session
        """
        async with cls.ctx.get_async_session(ro=ro, retries=retries, retry_interval=retry_interval, session=session, **kwargs) as sess:
            yield sess

    def create_all(cls, base: Optional[Any]=None):
        """
        Creates all tables
        """
        base = base or Base
        return cls.ctx.create_all(base=base)

    def drop_all(cls, base: Optional[Any]=None):
        """
        Drops all tables
        """
        base = base or Base
        return cls.ctx.drop_all(base=base)

    async def async_create_all(cls, base: Optional[Any]=None):
        """
        Creates all tables
        """
        base = base or Base
        return await cls.ctx.async_create_all(base=base)

    async def async_drop_all(cls, base: Optional[Any]=None):
        """
        Drops all tables
        """
        base = base or Base
        return await cls.ctx.async_drop_all(base=base)