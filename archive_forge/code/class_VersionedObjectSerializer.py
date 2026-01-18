import abc
import collections
from collections import abc as collections_abc
import copy
import functools
import logging
import warnings
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils as vutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields as obj_fields
class VersionedObjectSerializer(messaging.NoOpSerializer):
    """A VersionedObject-aware Serializer.

    This implements the Oslo Serializer interface and provides the
    ability to serialize and deserialize VersionedObject entities. Any service
    that needs to accept or return VersionedObjects as arguments or result
    values should pass this to its RPCClient and RPCServer objects.
    """
    OBJ_BASE_CLASS = VersionedObject

    def _do_backport(self, context, objprim, objclass):
        obj_versions = obj_tree_get_versions(objclass.obj_name())
        indirection_api = self.OBJ_BASE_CLASS.indirection_api
        try:
            return indirection_api.object_backport_versions(context, objprim, obj_versions)
        except NotImplementedError:
            return indirection_api.object_backport(context, objprim, objclass.VERSION)

    def _process_object(self, context, objprim):
        try:
            return self.OBJ_BASE_CLASS.obj_from_primitive(objprim, context=context)
        except exception.IncompatibleObjectVersion:
            with excutils.save_and_reraise_exception(reraise=False) as ctxt:
                verkey = '%s.version' % self.OBJ_BASE_CLASS.OBJ_SERIAL_NAMESPACE
                objver = objprim[verkey]
                if objver.count('.') == 2:
                    objprim[verkey] = '.'.join(objver.split('.')[:2])
                    return self._process_object(context, objprim)
                namekey = '%s.name' % self.OBJ_BASE_CLASS.OBJ_SERIAL_NAMESPACE
                objname = objprim[namekey]
                supported = VersionedObjectRegistry.obj_classes().get(objname, [])
                if self.OBJ_BASE_CLASS.indirection_api and supported:
                    return self._do_backport(context, objprim, supported[0])
                else:
                    ctxt.reraise = True

    def _process_iterable(self, context, action_fn, values):
        """Process an iterable, taking an action on each value.

        :param:context: Request context
        :param:action_fn: Action to take on each item in values
        :param:values: Iterable container of things to take action on
        :returns: A new container of the same type (except set) with
                  items from values having had action applied.
        """
        iterable = values.__class__
        if issubclass(iterable, dict):
            return iterable([(k, action_fn(context, v)) for k, v in values.items()])
        else:
            if iterable == set:
                iterable = list
            return iterable([action_fn(context, value) for value in values])

    def serialize_entity(self, context, entity):
        if isinstance(entity, (tuple, list, set, dict)):
            entity = self._process_iterable(context, self.serialize_entity, entity)
        elif hasattr(entity, 'obj_to_primitive') and callable(entity.obj_to_primitive):
            entity = entity.obj_to_primitive()
        return entity

    def deserialize_entity(self, context, entity):
        namekey = '%s.name' % self.OBJ_BASE_CLASS.OBJ_SERIAL_NAMESPACE
        if isinstance(entity, dict) and namekey in entity:
            entity = self._process_object(context, entity)
        elif isinstance(entity, (tuple, list, set, dict)):
            entity = self._process_iterable(context, self.deserialize_entity, entity)
        return entity