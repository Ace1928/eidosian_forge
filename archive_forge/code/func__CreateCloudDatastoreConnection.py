from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def _CreateCloudDatastoreConnection(connection_fn, app_id, external_app_ids, kwargs):
    """Creates a new context to connect to a remote Cloud Datastore instance.

  This should only be used outside of Google App Engine.

  Args:
    connection_fn: A connection function which accepts both an _api_version
      and an _id_resolver argument.
    app_id: The application id to connect to. This differs from the project
      id as it may have an additional prefix, e.g. "s~" or "e~".
    external_app_ids: A list of apps that may be referenced by data in your
      application. For example, if you are connected to s~my-app and store keys
      for s~my-other-app, you should include s~my-other-app in the external_apps
      list.
    kwargs: The additional kwargs to pass to the connection_fn.

  Raises:
    ValueError: if the app_id provided doesn't match the current environment's
        APPLICATION_ID.

  Returns:
    An ndb.Context that can connect to a Remote Cloud Datastore. You can use
    this context by passing it to ndb.set_context.
  """
    from googlecloudsdk.third_party.appengine.datastore import cloud_datastore_v1_remote_stub
    if not datastore_pbs._CLOUD_DATASTORE_ENABLED:
        raise datastore_errors.BadArgumentError(datastore_pbs.MISSING_CLOUD_DATASTORE_MESSAGE)
    current_app_id = os.environ.get('APPLICATION_ID', None)
    if current_app_id and current_app_id != app_id:
        raise ValueError('Cannot create a Cloud Datastore context that connects to an application (%s) that differs from the application already connected to (%s).' % (app_id, current_app_id))
    os.environ['APPLICATION_ID'] = app_id
    id_resolver = datastore_pbs.IdResolver((app_id,) + tuple(external_app_ids))
    project_id = id_resolver.resolve_project_id(app_id)
    endpoint = googledatastore.helper.get_project_endpoint_from_env(project_id)
    datastore = googledatastore.Datastore(project_endpoint=endpoint, credentials=googledatastore.helper.get_credentials_from_env())
    kwargs['_api_version'] = _CLOUD_DATASTORE_V1
    kwargs['_id_resolver'] = id_resolver
    conn = connection_fn(**kwargs)
    try:
        stub = cloud_datastore_v1_remote_stub.CloudDatastoreV1RemoteStub(datastore)
        apiproxy_stub_map.apiproxy.RegisterStub(_CLOUD_DATASTORE_V1, stub)
    except:
        pass
    try:
        apiproxy_stub_map.apiproxy.RegisterStub('memcache', _ThrowingStub())
    except:
        pass
    try:
        apiproxy_stub_map.apiproxy.RegisterStub('taskqueue', _ThrowingStub())
    except:
        pass
    return conn