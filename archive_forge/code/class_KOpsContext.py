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
class KOpsContext:

    def __init__(self, settings: KOpsSettings, ctx: Optional[str]=None, name: Optional[str]=None, config_file: Optional[str]=None):
        self.settings = settings
        self.ctx = ctx
        self.name = name
        self.config_file = config_file
        self.ainitialized: bool = False
        self.initialized: bool = False

    async def aset_k8_config(self):
        if self.ainitialized:
            return
        if self.settings.in_k8s:
            logger.info('Loading in-cluster config')
            AsyncConfig.load_incluster_config()
        else:
            config = self.settings.get_kconfig_path(self.ctx or self.name)
            if config:
                config = config.as_posix()
            logger.info(f'Loading kubeconfig from {config}')
            await AsyncConfig.load_kube_config(config_file=config, context=self.ctx or self.name or self.settings.kubeconfig_context)
        logger.info('Initialized K8s Client')
        self.ainitialized = True

    def set_k8_config(self):
        if self.initialized:
            return
        if self.settings.in_k8s:
            logger.info('Loading in-cluster config')
            SyncConfig.load_incluster_config()
        else:
            config = self.settings.get_kconfig_path(self.ctx or self.name)
            if config:
                config = config.as_posix()
            logger.info(f'Loading kubeconfig from {config}')
            SyncConfig.load_kube_config(config_file=config, context=self.ctx or self.name or self.settings.kubeconfig_context)
        logger.info('Initialized K8s Client')
        self.initialized = True

    @lazyproperty
    def client(self) -> 'SyncClient.ApiClient':
        """
        Primary Sync Client
        """
        return SyncClient.ApiClient(pool_threads=4)

    @lazyproperty
    def aclient(self) -> 'AsyncClient.ApiClient':
        """
        Primary Async Client
        """
        return AsyncClient.ApiClient(pool_threads=4)

    @lazyproperty
    def wsclient(self) -> 'SyncStream.WSClient':
        """
        Primary Websocket Client
        """
        return SyncStream.WSClient(pool_threads=4)

    @lazyproperty
    def awsclient(self) -> 'AsyncStream.WsApiClient':
        """
        Primary Async Websocket Client
        """
        return AsyncStream.WsApiClient()

    @lazyproperty
    def core_v1(self) -> 'SyncClient.CoreV1Api':
        """
        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        - Secrets
        - Pods
        - Nodes
        """
        return SyncClient.CoreV1Api(self.client)

    @lazyproperty
    def core_v1_ws(self) -> 'SyncClient.CoreV1Api':
        """
        Websocket Client

        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        - Secrets
        - Pods
        - Nodes
        """
        return SyncClient.CoreV1Api(self.wsclient)

    @lazyproperty
    def apps_v1(self) -> 'SyncClient.AppsV1Api':
        """
        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        """
        return SyncClient.AppsV1Api(self.client)

    @lazyproperty
    def networking_v1(self) -> 'SyncClient.NetworkingV1Api':
        """
        - Ingress
        """
        return SyncClient.NetworkingV1Api(self.client)

    @lazyproperty
    def crds(self) -> 'SyncClient.ApiextensionsV1Api':
        return SyncClient.ApiextensionsV1Api(self.client)

    @lazyproperty
    def customobjs(self) -> 'SyncClient.CustomObjectsApi':
        return SyncClient.CustomObjectsApi(self.client)
    '\n    Async Properties\n    '

    @lazyproperty
    def acore_v1(self) -> 'AsyncClient.CoreV1Api':
        """
        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        - Secrets
        - Pods
        - Nodes
        """
        return AsyncClient.CoreV1Api(self.aclient)

    @lazyproperty
    def acore_v1_ws(self) -> 'AsyncClient.CoreV1Api':
        """
        Websocket Client

        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        - Secrets
        - Pods
        - Nodes
        """
        return AsyncClient.CoreV1Api(self.awsclient)

    @lazyproperty
    def aapps_v1(self) -> 'AsyncClient.AppsV1Api':
        """
        - StatefulSets
        - Deployments
        - DaemonSets
        - ReplicaSets
        """
        return AsyncClient.AppsV1Api(self.aclient)

    @lazyproperty
    def anetworking_v1(self) -> 'AsyncClient.NetworkingV1Api':
        """
        - Ingress
        """
        return AsyncClient.NetworkingV1Api(self.aclient)

    @lazyproperty
    def acrds(self) -> 'AsyncClient.ApiextensionsV1Api':
        return AsyncClient.ApiextensionsV1Api(self.aclient)

    @lazyproperty
    def acustomobjs(self) -> 'AsyncClient.CustomObjectsApi':
        return AsyncClient.CustomObjectsApi(self.aclient)
    '\n    Sync Resource Level\n    '

    @lazyproperty
    def config_maps(self) -> 'SyncClient.CoreV1Api':
        return self.core_v1

    @lazyproperty
    def secrets(self) -> 'SyncClient.CoreV1Api':
        return self.core_v1

    @lazyproperty
    def pods(self) -> 'SyncClient.CoreV1Api':
        return self.core_v1

    @lazyproperty
    def nodes(self) -> 'SyncClient.CoreV1Api':
        return self.core_v1

    @lazyproperty
    def services(self) -> 'SyncClient.CoreV1Api':
        return self.core_v1

    @lazyproperty
    def ingresses(self) -> 'SyncClient.NetworkingV1Api':
        return self.networking_v1

    @lazyproperty
    def stateful_sets(self) -> 'SyncClient.AppsV1Api':
        return self.apps_v1

    @lazyproperty
    def deployments(self) -> 'SyncClient.AppsV1Api':
        return self.apps_v1

    @lazyproperty
    def daemon_sets(self) -> 'SyncClient.AppsV1Api':
        return self.apps_v1

    @lazyproperty
    def replica_sets(self) -> 'SyncClient.AppsV1Api':
        return self.apps_v1

    @lazyproperty
    def customresourcedefinitions(self) -> 'SyncClient.ApiextensionsV1Api':
        return self.crds

    @lazyproperty
    def customobjects(self) -> 'SyncClient.CustomObjectsApi':
        return self.customobjs

    @lazyproperty
    def persistent_volumes(self) -> 'SyncClient.CoreV1Api':
        return self.core_v1

    @lazyproperty
    def persistent_volume_claims(self) -> 'SyncClient.CoreV1Api':
        return self.core_v1
    '\n    Async Resource Level\n    '

    @lazyproperty
    def aconfig_maps(self) -> 'AsyncClient.CoreV1Api':
        return self.acore_v1

    @lazyproperty
    def asecrets(self) -> 'AsyncClient.CoreV1Api':
        return self.acore_v1

    @lazyproperty
    def apods(self) -> 'AsyncClient.CoreV1Api':
        return self.acore_v1

    @lazyproperty
    def anodes(self) -> 'AsyncClient.CoreV1Api':
        return self.acore_v1

    @lazyproperty
    def aservices(self) -> 'AsyncClient.CoreV1Api':
        return self.acore_v1

    @lazyproperty
    def aingresses(self) -> 'AsyncClient.NetworkingV1Api':
        return self.anetworking_v1

    @lazyproperty
    def astateful_sets(self) -> 'AsyncClient.AppsV1Api':
        return self.aapps_v1

    @lazyproperty
    def adeployments(self) -> 'AsyncClient.AppsV1Api':
        return self.aapps_v1

    @lazyproperty
    def adaemon_sets(self) -> 'AsyncClient.AppsV1Api':
        return self.aapps_v1

    @lazyproperty
    def areplica_sets(self) -> 'AsyncClient.AppsV1Api':
        return self.aapps_v1

    @lazyproperty
    def acustomresourcedefinitions(self) -> 'AsyncClient.ApiextensionsV1Api':
        return self.acrds

    @lazyproperty
    def acustomobjects(self) -> 'AsyncClient.CustomObjectsApi':
        return self.acustomobjs

    @lazyproperty
    def apersistent_volumes(self) -> 'AsyncClient.CoreV1Api':
        return self.acore_v1

    @lazyproperty
    def apersistent_volume_claims(self) -> 'AsyncClient.CoreV1Api':
        return self.acore_v1

    @property
    def auth_headers(self) -> Dict[str, str]:
        if not self.initialized and (not self.ainitialized):
            return None
        auth = self.aclient.configuration.auth_settings() if self.ainitialized else self.client.configuration.auth_settings()
        return {auth['BearerToken']['key']: auth['BearerToken']['value']}

    @property
    def request_headers(self) -> Dict[str, str]:
        if not self.initialized and (not self.ainitialized):
            return None
        headers = self.aclient.default_headers if self.ainitialized else self.client.default_headers
        return {'Accept': 'application/json;as=Table;v=v1;g=meta.k8s.io,application/json;as=Table;v=v1beta1;g=meta.k8s.io,application/json', 'Connection': 'keep-alive', **self.auth_headers, **headers}

    @property
    def cluster_url(self) -> str:
        if self.initialized or self.ainitialized:
            return self.aclient.configuration.host if self.ainitialized else self.client.configuration.host
        else:
            return None

    @property
    def ssl_ca_cert(self) -> str:
        if self.initialized or self.ainitialized:
            return self.aclient.configuration.ssl_ca_cert if self.ainitialized else self.client.configuration.ssl_ca_cert
        else:
            return None

    @lazyproperty
    def http_client(self) -> Union['aiohttpx.Client', 'requests.Session']:
        """
        Returns the http client
        """
        with contextlib.suppress(ImportError):
            import aiohttpx
            return aiohttpx.Client(base_url=self.cluster_url, headers=self.request_headers, verify=self.ssl_ca_cert)
        with contextlib.suppress(ImportError):
            import requests
            sess = requests.Session()
            sess.headers.update(self.request_headers)
            sess.verify = self.ssl_ca_cert
            return sess
        raise ImportError('No http client found. Please install aiohttpx or requests')

    @staticmethod
    def to_singular(resource: str) -> str:
        if resource.endswith('es'):
            return resource[:-2]
        elif resource.endswith('s'):
            return resource[:-1]
        return resource
    '\n    API Methods\n    '

    def get(self, resource: str, name: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        api = getattr(self, resource)
        singular = self.to_singular(resource)
        return getattr(api, f'read_namespaced_{singular}')(name, namespace=namespace, **kwargs)

    def list(self, resource: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        api = getattr(self, resource)
        singular = self.to_singular(resource)
        if namespace:
            return getattr(api, f'list_namespaced_{singular}')(namespace=namespace, **kwargs)
        return getattr(api, f'list_{singular}_for_all_namespaces')(**kwargs)

    def create(self, resource: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        api = getattr(self, resource)
        singular = self.to_singular(resource)
        return getattr(api, f'create_namespaced_{singular}')(namespace=namespace, **kwargs)

    def update(self, resource: str, name: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        api = getattr(self, resource)
        singular = self.to_singular(resource)
        return getattr(api, f'patch_namespaced_{singular}')(name, namespace=namespace, **kwargs)

    def delete(self, resource: str, name: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        api = getattr(self, resource)
        singular = self.to_singular(resource)
        return getattr(api, f'delete_namespaced_{singular}')(name, namespace=namespace, **kwargs)

    def patch(self, resource: str, name: str, namespace: str=None, **kwargs) -> 'SyncClient.V1ObjectMeta':
        api = getattr(self, resource)
        singular = self.to_singular(resource)
        return getattr(api, f'patch_namespaced_{singular}')(name, namespace=namespace, **kwargs)

    async def aget(self, resource: str, name: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        aresource = resource if resource.startswith('a') else f'a{resource}'
        api = getattr(self, aresource)
        singular = self.to_singular(resource)
        return await getattr(api, f'read_namespaced_{singular}')(name, namespace=namespace, **kwargs)

    async def alist(self, resource: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        aresource = resource if resource.startswith('a') else f'a{resource}'
        api = getattr(self, aresource)
        singular = self.to_singular(resource)
        if namespace:
            return await getattr(api, f'list_namespaced_{singular}')(namespace=namespace, **kwargs)
        return await getattr(api, f'list_{singular}_for_all_namespaces')(**kwargs)

    async def acreate(self, resource: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        aresource = resource if resource.startswith('a') else f'a{resource}'
        api = getattr(self, aresource)
        singular = self.to_singular(resource)
        return await getattr(api, f'create_namespaced_{singular}')(namespace=namespace, **kwargs)

    async def aupdate(self, resource: str, name: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        aresource = resource if resource.startswith('a') else f'a{resource}'
        api = getattr(self, aresource)
        singular = self.to_singular(resource)
        return await getattr(api, f'patch_namespaced_{singular}')(name, namespace=namespace, **kwargs)

    async def adelete(self, resource: str, name: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        aresource = resource if resource.startswith('a') else f'a{resource}'
        api = getattr(self, aresource)
        singular = self.to_singular(resource)
        return await getattr(api, f'delete_namespaced_{singular}')(name, namespace=namespace, **kwargs)

    async def apatch(self, resource: str, name: str, namespace: str=None, **kwargs) -> 'AsyncClient.V1ObjectMeta':
        aresource = resource if resource.startswith('a') else f'a{resource}'
        api = getattr(self, aresource)
        singular = self.to_singular(resource)
        return await getattr(api, f'patch_namespaced_{singular}')(name, namespace=namespace, **kwargs)