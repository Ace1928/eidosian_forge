from __future__ import annotations
import random
import asyncio
import logging
import contextlib
from enum import Enum
from lazyops.libs.kops.base import *
from lazyops.libs.kops.config import KOpsSettings
from lazyops.libs.kops.utils import cached, DillSerializer, SignalHandler
from lazyops.libs.kops._kopf import kopf
from lazyops.types import lazyproperty
from lazyops.utils import logger
from typing import List, Dict, Union, Any, Optional, Callable, TYPE_CHECKING
import lazyops.libs.kops.types as t
import lazyops.libs.kops.atypes as at
class KOpsClientMeta(type):
    """
    This is the metaclass for the KOpsClient class.
    """
    ctx: Optional[str] = None
    _settings: Optional[KOpsSettings] = None
    _session: Optional[KOpsContext] = None
    _sessions: Dict[str, KOpsContext] = {}
    _startup_functions: List[Callable] = []
    _shutdown_functions: List[Callable] = []

    @property
    def settings(cls) -> KOpsSettings:
        if not cls._settings:
            cls._settings = KOpsSettings()
        return cls._settings

    def add_session(cls, name: Optional[str]=None, config_file: Optional[str]=None, ctx: Optional[Any]=None, set_as_current: Optional[bool]=True, **kwargs):
        """
        Adds a session to the client.
        """
        name = name or 'default'
        if name not in cls._sessions:
            cls._sessions[name] = KOpsContext(settings=cls.settings, config_file=config_file, ctx=ctx, **kwargs)
            logger.info(f'Session created: {name}')
        if set_as_current:
            cls.ctx = name
            cls._session = cls._sessions[name]
            logger.info(f'Session set: {name}')

    def set_session(cls, name: Optional[str]=None):
        """
        Sets the current session.
        """
        name = name or 'default'
        if name in cls._sessions:
            cls.ctx = name
            cls._session = cls._sessions[name]
            logger.info(f'Session set: {name}')

    @property
    def session(cls) -> KOpsContext:
        if not cls._session:
            cls._session = KOpsContext(settings=cls.settings, ctx=cls.ctx)
            if cls.ctx is None:
                cls.ctx = 'default'
            cls._sessions[cls.ctx] = cls._session
            logger.info(f'Session set: {cls.ctx}')
        return cls._session

    @property
    def client(cls) -> 'SyncClient.ApiClient':
        """
        Returns the kubernetes client.
        """
        return cls.session.client

    @property
    def aclient(cls) -> 'AsyncClient.ApiClient':
        """
        Returns the async kubernetes client.
        """
        return cls.session.aclient
    '\n    HTTP Client Properties\n    '

    @property
    def auth_headers(cls) -> Dict[str, str]:
        """
        Returns the authentication headers.
        """
        return cls.session.auth_headers

    @property
    def request_headers(cls) -> Dict[str, str]:
        """
        Returns the request headers.
        """
        return cls.session.request_headers

    @property
    def cluster_url(cls) -> str:
        """
        Returns the cluster url.
        """
        return cls.session.cluster_url

    @property
    def ssl_ca_cert(cls) -> str:
        """
        Returns the ssl ca cert.
        """
        return cls.session.ssl_ca_cert

    @property
    def http_client(cls) -> Union['aiohttpx.Client', 'requests.Session']:
        """
        Returns the http client.
        """
        return cls.session.http_client

    @property
    def wsclient(cls) -> 'SyncStream.WSClient':
        """
        Returns the websocket kubernetes client.
        """
        return cls.session.wsclient

    @property
    def awsclient(cls) -> 'AsyncStream.WsApiClient':
        """
        Returns the async websocket kubernetes client.
        """
        return cls.session.awsclient

    @property
    def apps_v1(cls) -> 'SyncClient.AppsV1Api':
        """
        Returns the apps_v1 api.
        """
        return cls.session.apps_v1

    @property
    def core_v1(cls) -> 'SyncClient.CoreV1Api':
        """
        Returns the core_v1 api.
        """
        return cls.session.core_v1

    @property
    def core_v1_ws(cls) -> 'SyncClient.CoreV1Api':
        """
        Websocket Client:
        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        - Secrets
        - Pods
        - Nodes
        """
        return cls.session.core_v1_ws

    @property
    def crds(cls) -> 'SyncClient.CustomObjectsApi':
        """
        Returns the crds_v1 api.
        """
        return cls.session.crds

    @property
    def customobjs(cls) -> 'SyncClient.CustomObjectsApi':
        return cls.session.customobjs
    '\n    Async Properties\n    '

    @property
    def acore_v1(cls) -> 'AsyncClient.CoreV1Api':
        """
        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        - Secrets
        - Pods
        - Nodes
        """
        return cls.session.acore_v1

    @property
    def acore_v1_ws(cls) -> 'AsyncClient.CoreV1Api':
        """
        Websocket Client:
        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        - Secrets
        - Pods
        - Nodes
        """
        return cls.session.acore_v1_ws

    @property
    def aapps_v1(cls) -> 'AsyncClient.AppsV1Api':
        """
        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        """
        return cls.session.aapps_v1

    @property
    def anetworking_v1(cls) -> 'AsyncClient.NetworkingV1Api':
        """
        - Ingress
        """
        return cls.session.anetworking_v1

    @property
    def acrds(cls) -> 'AsyncClient.ApiextensionsV1Api':
        return cls.session.acrds

    @property
    def acustomobjs(cls) -> 'AsyncClient.CustomObjectsApi':
        return cls.session.acustomobjs
    '\n    Sync Resource Level\n    '

    @property
    def config_maps(cls) -> 'SyncClient.CoreV1Api':
        return cls.session.config_maps

    @property
    def secrets(cls) -> 'SyncClient.CoreV1Api':
        return cls.session.secrets

    @property
    def pods(cls) -> 'SyncClient.CoreV1Api':
        return cls.session.pods

    @property
    def nodes(cls) -> 'SyncClient.CoreV1Api':
        return cls.session.nodes

    @property
    def services(cls) -> 'SyncClient.CoreV1Api':
        return cls.session.services

    @property
    def ingresses(cls) -> 'SyncClient.NetworkingV1Api':
        return cls.session.ingresses

    @property
    def stateful_sets(cls) -> 'SyncClient.AppsV1Api':
        return cls.session.stateful_sets

    @property
    def deployments(cls) -> 'SyncClient.AppsV1Api':
        return cls.session.deployments

    @property
    def daemon_sets(cls) -> 'SyncClient.AppsV1Api':
        return cls.session.daemon_sets

    @property
    def replica_sets(cls) -> 'SyncClient.AppsV1Api':
        return cls.session.replica_sets

    @property
    def customresourcedefinitions(cls) -> 'SyncClient.ApiextensionsV1Api':
        return cls.session.customresourcedefinitions

    @property
    def customobjects(cls) -> 'SyncClient.CustomObjectsApi':
        return cls.session.customobjects

    @property
    def persistent_volumes(cls) -> 'SyncClient.CoreV1Api':
        return cls.session.persistent_volumes

    @property
    def persistent_volume_claims(cls) -> 'SyncClient.CoreV1Api':
        return cls.session.persistent_volume_claims
    '\n    Async Resource Level\n    '

    @property
    def aconfig_maps(cls) -> 'AsyncClient.CoreV1Api':
        return cls.session.aconfig_maps

    @property
    def asecrets(cls) -> 'AsyncClient.CoreV1Api':
        return cls.session.asecrets

    @property
    def apods(cls) -> 'AsyncClient.CoreV1Api':
        return cls.session.apods

    @property
    def anodes(cls) -> 'AsyncClient.CoreV1Api':
        return cls.session.anodes

    @property
    def aservices(cls) -> 'AsyncClient.CoreV1Api':
        return cls.session.aservices

    @property
    def aingresses(cls) -> 'AsyncClient.NetworkingV1Api':
        return cls.session.aingresses

    @property
    def astateful_sets(cls) -> 'AsyncClient.AppsV1Api':
        return cls.session.astateful_sets

    @property
    def adeployments(cls) -> 'AsyncClient.AppsV1Api':
        return cls.session.adeployments

    @property
    def adaemon_sets(cls) -> 'AsyncClient.AppsV1Api':
        return cls.session.adaemon_sets

    @property
    def areplica_sets(cls) -> 'AsyncClient.AppsV1Api':
        return cls.session.areplica_sets

    @property
    def acustomresourcedefinitions(cls) -> 'AsyncClient.ApiextensionsV1Api':
        return cls.session.acustomresourcedefinitions

    @property
    def acustomobjects(cls) -> 'AsyncClient.CustomObjectsApi':
        return cls.session.acustomobjects

    @property
    def apersistent_volumes(cls) -> 'AsyncClient.CoreV1Api':
        return cls.session.apersistent_volumes

    @property
    def apersistent_volume_claims(cls) -> 'AsyncClient.CoreV1Api':
        return cls.session.apersistent_volume_claims

    async def aconfigure(cls):
        """
        Sets the kubernetes config for the current context.
        """
        await cls.session.aset_k8_config()

    async def aset_k8_config(cls):
        """
        Sets the kubernetes config for the current context.
        """
        await cls.session.aset_k8_config()

    def configure(cls):
        """
        Sets the kubernetes config for the current context.
        """
        cls.session.set_k8_config()

    def set_k8_config(cls):
        """
        Sets the kubernetes config for the current context.
        """
        cls.session.set_k8_config()

    def get(cls, resource: str, name: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        """
        Gets a resource.
        """
        return cls.session.get(resource, name, namespace, **kwargs)

    def list(cls, resource: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        """
        Lists a resource.
        """
        return cls.session.list(resource, namespace, **kwargs)

    def create(cls, resource: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        """
        Creates a resource.
        """
        return cls.session.create(resource, namespace, **kwargs)

    def update(cls, resource: str, name: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        """
        Updates a resource.
        """
        return cls.session.update(resource, name, namespace, **kwargs)

    def delete(cls, resource: str, name: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        """
        Deletes a resource.
        """
        return cls.session.delete(resource, name, namespace, **kwargs)

    def patch(cls, resource: str, name: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        """
        Patches a resource.
        """
        return cls.session.patch(resource, name, namespace, **kwargs)

    async def aget(cls, resource: str, name: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        """
        Gets a resource.
        """
        return await cls.session.aget(resource, name, namespace, **kwargs)

    async def alist(cls, resource: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        """
        Lists a resource.
        """
        return await cls.session.alist(resource, namespace, **kwargs)

    async def acreate(cls, resource: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        """
        Creates a resource.
        """
        return await cls.session.acreate(resource, namespace, **kwargs)

    async def aupdate(cls, resource: str, name: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        """
        Updates a resource.
        """
        return await cls.session.aupdate(resource, name, namespace, **kwargs)

    async def adelete(cls, resource: str, name: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        """
        Deletes a resource.
        """
        return await cls.session.adelete(resource, name, namespace, **kwargs)

    async def apatch(cls, resource: str, name: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        """
        Patches a resource.
        """
        return await cls.session.apatch(resource, name, namespace, **kwargs)
    '\n    KOpf Methods\n    '

    def add_function(cls, function: Callable, event: EventType=EventType.startup):
        if event == EventType.startup:
            cls._startup_functions.append(function)
        elif event == EventType.shutdown:
            cls._shutdown_functions.append(function)

    async def run_startup_functions(cls, **kwargs):
        asyncio.create_task(SignalHandler.monitor(cls._shutdown_functions))
        if not cls._startup_functions:
            return
        for func in cls._startup_functions:
            await func(**kwargs)

    async def run_shutdown_functions(cls, **kwargs):
        if not cls._shutdown_functions:
            return
        for func in cls._shutdown_functions:
            await func(**kwargs)

    def configure_kopf(cls, _logger: logging.Logger=None, _settings: Optional[kopf.OperatorSettings]=None, enable_event_logging: Optional[bool]=None, event_logging_level: Optional[str]=None, finalizer: Optional[str]=None, storage_prefix: Optional[str]=None, persistent_key: Optional[str]=None, error_delays: Optional[List[int]]=None, startup_functions: Optional[List[Callable]]=None, shutdown_functions: Optional[List[Callable]]=None, kopf_name: Optional[str]=None, app_name: Optional[str]=None, multi_pods: Optional[bool]=True, **kwargs):
        """
        Registers the startup function for configuring kopf.

        Parameters
        """
        enable_event_logging = enable_event_logging if enable_event_logging is not None else cls.settings.kopf_enable_event_logging
        event_logging_level = event_logging_level if event_logging_level is not None else cls.settings.kopf_event_logging_level
        finalizer = finalizer if finalizer is not None else cls.settings.kops_finalizer
        if finalizer != cls.settings.kops_finalizer:
            cls.settings.kops_finalizer = finalizer
        storage_prefix = storage_prefix if storage_prefix is not None else cls.settings.kops_prefix
        persistent_key = persistent_key if persistent_key is not None else cls.settings.kops_persistent_key
        if persistent_key != cls.settings.kops_persistent_key:
            cls.settings.kops_persistent_key = persistent_key
        error_delays = error_delays if error_delays is not None else [10, 20, 30]
        kopf_name = kopf_name if kopf_name is not None else cls.settings.kopf_name
        app_name = app_name if app_name is not None else cls.settings.app_name
        _logger = _logger if _logger is not None else logger
        if startup_functions is not None:
            for func in startup_functions:
                cls.add_function(func, EventType.startup)
        if shutdown_functions is not None:
            for func in shutdown_functions:
                cls.add_function(func, EventType.shutdown)

        @kopf.on.startup()
        async def configure(settings: kopf.OperatorSettings, **kwargs):
            if _settings is not None:
                settings = _settings
            if enable_event_logging is False:
                settings.posting.enabled = enable_event_logging
                _logger.info(f'Kopf Events Enabled: {enable_event_logging}')
            if event_logging_level is not None:
                settings.posting.level = logging.getLevelName(event_logging_level.upper())
                _logger.info(f'Kopf Events Logging Level: {event_logging_level}')
            settings.persistence.finalizer = finalizer
            settings.persistence.progress_storage = kopf.SmartProgressStorage(prefix=storage_prefix)
            settings.persistence.diffbase_storage = kopf.AnnotationsDiffBaseStorage(prefix=storage_prefix, key=persistent_key)
            settings.batching.error_delays = error_delays
            if multi_pods:
                settings.peering.priority = random.randint(0, 32767)
                settings.peering.stealth = True
                if kwargs.get('peering_name'):
                    settings.peering.name = kwargs.get('peering_name')
            _logger.info(f'Starting Kopf: {kopf_name} {app_name} @ {cls.settings.build_id}')
            await cls.aset_k8_config()
            if cls._startup_functions:
                _logger.info('Running Startup Functions')
                await cls.run_startup_functions()
            _logger.info('Completed Kopf Startup')

        @kopf.on.login()
        async def login_fn(**kwargs):
            return kopf.login_with_service_account(**kwargs) or kopf.login_with_kubeconfig(**kwargs) if cls.settings.in_k8s else kopf.login_via_client(**kwargs)